import { MLPWeights } from '../types';

// ═══════════════════════════════════════════════
// 浏览器端 MLP 训练器
// 网络结构: 24→32(LeakyReLU)→32(LeakyReLU)→12(Tanh)
// 与 mlpDecoupler.ts 完全一致
// ═══════════════════════════════════════════════

function leakyRelu(x: number): number {
  return x > 0 ? x : 0.01 * x;
}

function leakyReluDeriv(x: number): number {
  return x > 0 ? 1 : 0.01;
}

function tanhFn(x: number): number {
  return Math.tanh(x);
}

function tanhDeriv(x: number): number {
  const t = Math.tanh(x);
  return 1 - t * t;
}

// Xavier initialization
function xavierInit(rows: number, cols: number): number[][] {
  const scale = Math.sqrt(2.0 / (rows + cols));
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() - 0.5) * 2 * scale)
  );
}

function zeros(n: number): number[] {
  return Array(n).fill(0);
}

export interface TrainingConfig {
  epochs: number;
  learningRate: number;
  printInterval: number;
}

export interface TrainingProgress {
  epoch: number;
  trainLoss: number;
  valLoss: number;
}

export type ProgressCallback = (progress: TrainingProgress) => void;

export class MLPTrainer {
  // Layer weights and biases
  private w1: number[][];
  private b1: number[];
  private w2: number[][];
  private b2: number[];
  private w3: number[][];
  private b3: number[];

  constructor() {
    this.w1 = xavierInit(32, 24);
    this.b1 = zeros(32);
    this.w2 = xavierInit(32, 32);
    this.b2 = zeros(32);
    this.w3 = xavierInit(12, 32);
    this.b3 = zeros(12);
  }

  private forward(input: number[]): {
    output: number[];
    z1: number[]; a1: number[];
    z2: number[]; a2: number[];
    z3: number[]; a3: number[];
  } {
    // Layer 1: 24 → 32
    const z1: number[] = [];
    const a1: number[] = [];
    for (let i = 0; i < 32; i++) {
      let sum = this.b1[i];
      for (let j = 0; j < input.length; j++) {
        sum += this.w1[i][j] * input[j];
      }
      z1.push(sum);
      a1.push(leakyRelu(sum));
    }

    // Layer 2: 32 → 32
    const z2: number[] = [];
    const a2: number[] = [];
    for (let i = 0; i < 32; i++) {
      let sum = this.b2[i];
      for (let j = 0; j < 32; j++) {
        sum += this.w2[i][j] * a1[j];
      }
      z2.push(sum);
      a2.push(leakyRelu(sum));
    }

    // Layer 3: 32 → 12
    const z3: number[] = [];
    const a3: number[] = [];
    for (let i = 0; i < 12; i++) {
      let sum = this.b3[i];
      for (let j = 0; j < 32; j++) {
        sum += this.w3[i][j] * a2[j];
      }
      z3.push(sum);
      a3.push(tanhFn(sum));
    }

    return { output: a3, z1, a1, z2, a2, z3, a3 };
  }

  private backward(
    input: number[],
    target: number[],
    fwd: { z1: number[]; a1: number[]; z2: number[]; a2: number[]; z3: number[]; a3: number[] }
  ): {
    dw1: number[][]; db1: number[];
    dw2: number[][]; db2: number[];
    dw3: number[][]; db3: number[];
  } {
    const { z1, a1, z2, a2, z3, a3 } = fwd;

    // Output layer gradient (tanh derivative)
    const d3: number[] = [];
    for (let i = 0; i < 12; i++) {
      d3.push((a3[i] - target[i]) * tanhDeriv(z3[i]));
    }

    // Hidden layer 2 gradient
    const d2: number[] = [];
    for (let i = 0; i < 32; i++) {
      let sum = 0;
      for (let j = 0; j < 12; j++) {
        sum += this.w3[j][i] * d3[j];
      }
      d2.push(sum * leakyReluDeriv(z2[i]));
    }

    // Hidden layer 1 gradient
    const d1: number[] = [];
    for (let i = 0; i < 32; i++) {
      let sum = 0;
      for (let j = 0; j < 32; j++) {
        sum += this.w2[j][i] * d2[j];
      }
      d1.push(sum * leakyReluDeriv(z1[i]));
    }

    // Weight gradients
    const dw3 = d3.map(di => a2.map(aj => di * aj));
    const db3 = d3.slice();
    const dw2 = d2.map(di => a1.map(aj => di * aj));
    const db2 = d2.slice();
    const dw1 = d1.map(di => input.map(xj => di * xj));
    const db1 = d1.slice();

    return { dw1, db1, dw2, db2, dw3, db3 };
  }

  private applyGradients(
    grads: {
      dw1: number[][]; db1: number[];
      dw2: number[][]; db2: number[];
      dw3: number[][]; db3: number[];
    },
    lr: number
  ) {
    const { dw1, db1, dw2, db2, dw3, db3 } = grads;

    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 24; j++) this.w1[i][j] -= lr * dw1[i][j];
      this.b1[i] -= lr * db1[i];
    }
    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 32; j++) this.w2[i][j] -= lr * dw2[i][j];
      this.b2[i] -= lr * db2[i];
    }
    for (let i = 0; i < 12; i++) {
      for (let j = 0; j < 32; j++) this.w3[i][j] -= lr * dw3[i][j];
      this.b3[i] -= lr * db3[i];
    }
  }

  /**
   * 训练 MLP
   * @param inputs   归一化后的输入 (N×24)
   * @param targets  归一化后的目标 (N×12)
   * @param config   训练配置
   * @param onProgress 进度回调
   */
  async train(
    inputs: number[][],
    targets: number[][],
    config: TrainingConfig,
    onProgress?: ProgressCallback
  ): Promise<void> {
    const { epochs, learningRate, printInterval } = config;
    const N = inputs.length;
    const splitIdx = Math.floor(N * 0.8);

    const trainX = inputs.slice(0, splitIdx);
    const trainY = targets.slice(0, splitIdx);
    const valX = inputs.slice(splitIdx);
    const valY = targets.slice(splitIdx);

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle training data
      const indices = Array.from({ length: trainX.length }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      let trainLoss = 0;
      for (const idx of indices) {
        const fwd = this.forward(trainX[idx]);
        const grads = this.backward(trainX[idx], trainY[idx], fwd);
        this.applyGradients(grads, learningRate);

        for (let k = 0; k < 12; k++) {
          trainLoss += (fwd.a3[k] - trainY[idx][k]) ** 2;
        }
      }
      trainLoss /= (trainX.length * 12);

      if ((epoch + 1) % printInterval === 0 || epoch === epochs - 1) {
        let valLoss = 0;
        for (let i = 0; i < valX.length; i++) {
          const fwd = this.forward(valX[i]);
          for (let k = 0; k < 12; k++) {
            valLoss += (fwd.a3[k] - valY[i][k]) ** 2;
          }
        }
        valLoss /= (valX.length * 12);

        onProgress?.({ epoch: epoch + 1, trainLoss, valLoss });

        // Yield to UI thread
        await new Promise(r => setTimeout(r, 0));
      }
    }
  }

  /**
   * 导出为前端 mlpDecoupler 可用的权重格式
   */
  exportWeights(): MLPWeights {
    return {
      input: { weights: this.w1.map(r => [...r]), biases: [...this.b1] },
      hidden: { weights: this.w2.map(r => [...r]), biases: [...this.b2] },
      output: { weights: this.w3.map(r => [...r]), biases: [...this.b3] },
    };
  }
}

// ═══════════════════════════════════════════════
// 逐指独立运动的通道映射
// ═══════════════════════════════════════════════
export const FINGER_CHANNELS: Record<string, { label: string; channels: number[]; instruction: string }> = {
  thumb_tuck: {
    label: '拇指内扣',
    channels: [0, 1, 2],  // THUMB_IP, THUMB_MCP, THUMB_SPD
    instruction: '保持拇指指间关节伸直，整体向掌心内扣（做"点赞"手势的反向，拇指向掌心里收），其余手指保持伸直不动',
  },
  thumb_ip: {
    label: '拇指指间弯曲',
    channels: [0, 1, 2],  // THUMB_IP, THUMB_MCP, THUMB_SPD
    instruction: '保持拇指根部不动，只弯曲拇指最前端的指间关节（指尖向指根方向勾），其余手指保持伸直不动',
  },
  index: {
    label: '食指',
    channels: [3, 4, 5],  // INDEX_PIP, IM_SPD, INDEX_MCP
    instruction: '只弯曲/伸直食指，其余手指保持伸直不动',
  },
  middle: {
    label: '中指',
    channels: [6, 7, 8],  // MIDDLE_PIP, MR_SPD, MIDDLE_MCP
    instruction: '只弯曲/伸直中指，其余手指保持伸直不动',
  },
  ring_pinky: {
    label: '无名指+小指',
    channels: [9, 10, 11],  // RING_PIP, RP_SPD, PINKY_PIP
    instruction: '同时弯曲/伸直无名指和小指，其余手指保持伸直不动',
  },
  spread: {
    label: '五指张开',
    channels: [2, 4, 7, 10],  // THUMB_SPD, IM_SPD, MR_SPD, RP_SPD
    instruction: '五指并拢→尽力张开→并拢，重复动作',
  },
};

export const TRAINING_STEPS = ['rest', 'thumb_tuck', 'thumb_ip', 'index', 'middle', 'ring_pinky', 'spread'] as const;
export type TrainingStep = typeof TRAINING_STEPS[number];

/**
 * 从采集数据构造训练集（差值空间 + 伪标签）
 */
export function prepareTrainingData(
  segments: Record<string, number[][]>,
  baseline: number[]
): { inputs: number[][]; targets: number[][]; maxDelta: number[] } {
  // 合并所有段数据（差值空间）
  let allDeltas: number[][] = [];
  let allLabels: number[][] = [];

  for (const [finger, frames] of Object.entries(segments)) {
    if (finger === 'rest' || !FINGER_CHANNELS[finger]) continue;
    const activeCh = FINGER_CHANNELS[finger].channels;

    for (const frame of frames) {
      const delta = frame.map((v, i) => v - baseline[i]);
      const label = new Array(12).fill(0);
      for (const ch of activeCh) {
        label[ch] = delta[ch];
      }
      allDeltas.push(delta);
      allLabels.push(label);
    }
  }

  if (allDeltas.length === 0) {
    return { inputs: [], targets: [], maxDelta: new Array(12).fill(1) };
  }

  // 计算归一化参数
  const maxDelta = new Array(12).fill(0);
  for (const delta of allDeltas) {
    for (let i = 0; i < 12; i++) {
      maxDelta[i] = Math.max(maxDelta[i], Math.abs(delta[i]));
    }
  }
  for (let i = 0; i < 12; i++) maxDelta[i] = Math.max(maxDelta[i], 1);

  // 构造自回归输入 [delta_t, prev_output]
  const inputs: number[][] = [];
  const targets: number[][] = [];
  let prevOutput = new Array(12).fill(0);

  for (let i = 0; i < allDeltas.length; i++) {
    const normDelta = allDeltas[i].map((v, j) => v / maxDelta[j]);
    const normPrev = prevOutput.map((v, j) => v / maxDelta[j]);
    const normLabel = allLabels[i].map((v, j) => Math.max(-1, Math.min(1, v / maxDelta[j])));

    inputs.push([...normDelta, ...normPrev]);
    targets.push(normLabel);

    prevOutput = allLabels[i].slice();
  }

  return { inputs, targets, maxDelta };
}
