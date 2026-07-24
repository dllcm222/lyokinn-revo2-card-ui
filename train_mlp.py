#!/usr/bin/env python3
"""
MLP解耦器训练脚本 - 针对12通道液金传感器布局

传感器布局（CH编号 -> 数组索引）：
┌─────────────────────────────────────────────────────────────┐
│ 拇指 Thumb          │ 食指 Index         │ 中指 Middle     │
│ 1: THUMB_IP    [0]  │ 4: INDEX_PIP   [3]  │ 7: MIDDLE_PIP [6]│
│ 2: THUMB_MCP   [1]  │ 6: INDEX_MCP   [5]  │ 9: MIDDLE_MCP [8]│
│ 3: THUMB_SPD   [2]  │ 5: IM_SPD      [4]  │ 8: MR_SPD     [7]│
├─────────────────────────────────────────────────────────────┤
│ 无名指 Ring         │ 小指 Pinky          │ 张开传感器      │
│10: RING_PIP   [9]   │12: PINKY_PIP  [11]  │ 3,5,7,10 用于张开│
│11: RP_SPD    [10]   │                    │                │
└─────────────────────────────────────────────────────────────┘

采集流程（按顺序，每组10秒 @ 40Hz = 400帧）：
  1. 静止基线（前20帧保持不动）
  2. 拇指弯曲（只动拇指，重复握拳-松开）
  3. 食指弯曲（只动食指）
  4. 中指弯曲（只动中指）
  5. 无名指+小指弯曲（两指联动）
  6. 五指张开（spread）
  7. 混合动作（可选，用于增强泛化）

数据格式：CSV文件，每行12个整数，无表头
录制工具：使用App中的"开始录制"/"停止录制"按钮
"""

import numpy as np
import json
import torch
import torch.nn as nn
import argparse

# ══════════════════════════════════════════
# 传感器通道定义（CH编号 -> 数组索引）
# ══════════════════════════════════════════
CHANNEL_MAP = {
    # 拇指
    'THUMB_IP':     0,   # CH1
    'THUMB_MCP':    1,   # CH2  
    'THUMB_SPD':    2,   # CH3
    # 食指
    'INDEX_PIP':    3,   # CH4
    'IM_SPD':       4,   # CH5
    'INDEX_MCP':    5,   # CH6
    # 中指
    'MIDDLE_PIP':   6,   # CH7
    'MR_SPD':       7,   # CH8
    'MIDDLE_MCP':   8,   # CH9
    # 无名指
    'RING_PIP':     9,   # CH10
    'RP_SPD':      10,   # CH11
    # 小指
    'PINKY_PIP':   11,   # CH12
}

# 逐指独立运动的活跃通道组
FINGER_CHANNELS = {
    'thumb':       [CHANNEL_MAP['THUMB_IP'], CHANNEL_MAP['THUMB_MCP'], CHANNEL_MAP['THUMB_SPD']],
    'index':       [CHANNEL_MAP['INDEX_PIP'], CHANNEL_MAP['INDEX_MCP'], CHANNEL_MAP['IM_SPD']],
    'middle':      [CHANNEL_MAP['MIDDLE_PIP'], CHANNEL_MAP['MIDDLE_MCP'], CHANNEL_MAP['MR_SPD']],
    'ring_pinky':  [CHANNEL_MAP['RING_PIP'], CHANNEL_MAP['PINKY_PIP'], CHANNEL_MAP['RP_SPD']],
    'spread':      [CHANNEL_MAP['THUMB_SPD'], CHANNEL_MAP['IM_SPD'], CHANNEL_MAP['MR_SPD'], CHANNEL_MAP['RP_SPD']],
}

# ══════════════════════════════════════════
# 网络结构（与前端 mlpDecoupler.ts 一致）
# ══════════════════════════════════════════
class DecouplerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 32),      # [delta_t(12) + prev_output(12)] -> 32
            nn.LeakyReLU(0.01),
            nn.Linear(32, 32),      # 32 -> 32
            nn.LeakyReLU(0.01),
            nn.Linear(32, 12),      # 32 -> delta_output(12)
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.net(x)

# ══════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════
def main(args):
    print("=" * 60)
    print("MLP解耦器训练 - 液金传感器12CH")
    print("=" * 60)

    # 1. 加载原始数据
    print(f"\n[1/6] 加载数据: {args.input}")
    try:
        data = np.loadtxt(args.input, delimiter=",")
    except Exception as e:
        print(f"  错误: 无法加载文件 - {e}")
        return
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    if data.shape[1] != 12:
        print(f"  警告: 数据列数为 {data.shape[1]}，期望 12 列")
        if data.shape[1] > 12:
            data = data[:, :12]
            print(f"  已截断为前 12 列")
    
    print(f"  数据规模: {data.shape[0]} 帧 × {data.shape[1]} 通道")

    # 2. 计算静止基线（前若干帧的均值）
    print(f"\n[2/6] 计算静止基线（前 {args.baseline_frames} 帧）")
    baseline = data[:args.baseline_frames].mean(axis=0)
    print(f"  基线值: {['%.0f' % v for v in baseline]}")

    # 转换为差值空间
    deltas = data - baseline
    print(f"  最大差值范围: [{deltas.min():.0f}, {deltas.max():.0f}]")

    # 3. 自动分割动作段（基于帧范围）
    print(f"\n[3/6] 分割动作段")
    
    # 默认段定义（每组400帧 = 10秒 @ 40Hz）
    # 0-19: 基线
    # 20-419: 拇指
    # 420-819: 食指
    # 820-1219: 中指
    # 1220-1619: 无名指+小指
    # 1620-2019: 张开
    # 2020-2419: 混合（可选）
    frame_offset = args.baseline_frames
    segment_duration = args.segment_duration
    
    segments = [
        ('thumb',       frame_offset,              frame_offset + segment_duration),
        ('index',       frame_offset + segment_duration,      frame_offset + segment_duration * 2),
        ('middle',      frame_offset + segment_duration * 2,  frame_offset + segment_duration * 3),
        ('ring_pinky',  frame_offset + segment_duration * 3,  frame_offset + segment_duration * 4),
        ('spread',      frame_offset + segment_duration * 4,  frame_offset + segment_duration * 5),
    ]
    
    # 如果有足够数据，添加混合动作段
    if len(data) > frame_offset + segment_duration * 5:
        segments.append(('mixed', 
            frame_offset + segment_duration * 5,
            min(frame_offset + segment_duration * 6, len(data))))
    
    # 验证段范围
    segments = [(name, start, min(end, len(data))) 
                for name, start, end in segments if start < len(data)]
    
    print(f"  动作段配置:")
    total_frames = 0
    for name, start, end in segments:
        duration = (end - start) / 40.0
        print(f"    {name:10} : 帧 {start:5d}-{end:5d} ({duration:.1f}秒)")
        total_frames += (end - start)
    print(f"    {'总计':10} : {total_frames} 帧")

    # 4. 构造伪标签（差值空间）
    print(f"\n[4/6] 构造伪标签")
    label_deltas = np.zeros_like(deltas)
    
    for finger, start, end in segments:
        active_ch = FINGER_CHANNELS[finger]
        
        for i in range(start, end):
            # 非活跃通道差值为0（理想无耦合）
            label_deltas[i] = 0.0
            # 活跃通道保留真实差值
            label_deltas[i, active_ch] = deltas[i, active_ch]
        
        # 计算该段的耦合误差（非活跃通道的平均绝对差值）
        non_active_ch = [ch for ch in range(12) if ch not in active_ch]
        coupling_error = np.abs(deltas[start:end, non_active_ch]).mean()
        print(f"    {finger:10} : 耦合误差={coupling_error:.1f}")

    # 5. 构造自回归输入 [delta_t, prev_output]
    print(f"\n[5/6] 构造训练数据")
    N = len(deltas) - 1
    inputs = np.zeros((N, 24))
    targets = np.zeros((N, 12))
    
    prev_output = np.zeros(12)
    for i in range(N):
        inputs[i, :12] = deltas[i]
        inputs[i, 12:] = prev_output
        targets[i] = label_deltas[i]
        prev_output = label_deltas[i]  # 训练时用标签反馈
    
    # 归一化到 [-1, 1]（按通道独立缩放）
    max_delta = np.abs(deltas).max(axis=0) + 1e-6
    print(f"  各通道最大差值: {['%.0f' % v for v in max_delta]}")
    
    inputs_norm = np.zeros_like(inputs)
    for ch in range(12):
        inputs_norm[:, ch] = inputs[:, ch] / max_delta[ch]
        inputs_norm[:, ch + 12] = inputs[:, ch + 12] / max_delta[ch]
    
    targets_norm = targets / max_delta
    targets_norm = np.clip(targets_norm, -1, 1)
    
    # 分割训练/验证集
    split = int(len(inputs_norm) * 0.8)
    X_train = torch.FloatTensor(inputs_norm[:split])
    Y_train = torch.FloatTensor(targets_norm[:split])
    X_val = torch.FloatTensor(inputs_norm[split:])
    Y_val = torch.FloatTensor(targets_norm[split:])
    
    print(f"  训练样本: {len(X_train)}")
    print(f"  验证样本: {len(X_val)}")

    # 6. 训练
    print(f"\n[6/6] 训练网络")
    model = DecouplerMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % args.print_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), Y_val)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.output.replace('.json', '_best.pth'))
            
            print(f"  Epoch {epoch+1:4d}/{args.epochs}: "
                  f"train_loss={loss.item():.6f}  val_loss={val_loss.item():.6f}")
    
    # 7. 导出权重（前端格式）
    print(f"\n导出权重到 {args.output}")
    weights = {
        "input": {
            "weights": model.net[0].weight.detach().numpy().tolist(),
            "biases": model.net[0].bias.detach().numpy().tolist(),
        },
        "hidden": {
            "weights": model.net[2].weight.detach().numpy().tolist(),
            "biases": model.net[2].bias.detach().numpy().tolist(),
        },
        "output": {
            "weights": model.net[4].weight.detach().numpy().tolist(),
            "biases": model.net[4].bias.detach().numpy().tolist(),
        },
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=2)
    
    # 导出归一化参数
    norm_params = {
        "baseline": baseline.tolist(),
        "maxDelta": max_delta.tolist(),
    }
    norm_path = args.output.replace('.json', '_norm.json')
    with open(norm_path, 'w', encoding='utf-8') as f:
        json.dump(norm_params, f, indent=2)
    
    print(f"导出归一化参数到 {norm_path}")
    print("\n训练完成！")
    print(f"\n使用方法：")
    print(f"  1. 将 {args.output} 放置到项目 public 目录")
    print(f"  2. 在 App.tsx 中加载：")
    print(f"     const resp = await fetch('mlp-weights.json');")
    print(f"     mlpDecoupler.importWeights(await resp.text());")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP解耦器训练')
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', default='mlp-weights.json', help='输出权重文件路径')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--baseline-frames', type=int, default=20, help='基线帧数量')
    parser.add_argument('--segment-duration', type=int, default=400, help='每段帧数(400=10秒@40Hz)')
    parser.add_argument('--print-interval', type=int, default=50, help='打印间隔')
    
    args = parser.parse_args()
    main(args)
