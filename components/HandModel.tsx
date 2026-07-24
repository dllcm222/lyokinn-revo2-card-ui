
import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Group, MathUtils, Quaternion, Vector3 } from 'three';

// 预定义的旋转轴单位向量 (CMC 局部坐标系)
const AXIS_X = new Vector3(1, 0, 0);  // 屈伸轴 (FE, 位于大多角骨)
const AXIS_Y = new Vector3(0, 1, 0);  // 旋前/旋后轴 (PS, 绕骨长轴)
const AXIS_Z = new Vector3(0, 0, 1);  // 收展轴 (AA, 位于第1掌骨)
import { SensorReadings, SENSOR_MAP, HandSide, HandCalibration } from '../types';

const Bone = ({ radius, length, color = "#f7d5b8" }: { radius: number; length: number; color?: string }) => (
  <mesh position={[0, length / 2, 0]} castShadow receiveShadow>
    <capsuleGeometry args={[radius, length, 8, 16]} />
    <meshStandardMaterial 
      color={color} 
      roughness={0.8} 
      metalness={0.0} 
    />
  </mesh>
);

const Joint = ({ radius = 0.35, color = "#ff7e5f" }: { radius?: number; color?: string }) => (
  <mesh position={[0, 0, 0]} castShadow receiveShadow>
    <sphereGeometry args={[radius, 24, 24]} />
    <meshStandardMaterial 
      color={color} 
      roughness={0.3} 
      metalness={0.2}
    />
  </mesh>
);

interface HandModelProps {
  data: SensorReadings;
  calibration?: HandCalibration;
  side: HandSide;
  position?: [number, number, number];
}

export const HandModel: React.FC<HandModelProps> = ({ data, calibration, side, position = [0, -2, 0] }) => {
  const groupRef = useRef<Group>(null);
  // 对于掌心朝向用户的右手：由于我们是观测者，他人的右掌大拇指在观测者的右侧。
  // 基础模型大拇指在 x > 0 (右侧)，因此右手 mirror = 1
  const mirror = side === 'RIGHT' ? 1 : -1;
  const boneColor = "#f7d5b8"; // 温暖肤色
  const jointColor = "#ff7e5f"; // 明显的橙红色关节，便于识别

  // Finger Refs
  const thumbCMCRef = useRef<Group>(null);
  const thumbMCPRef = useRef<Group>(null);
  const thumbIPRef = useRef<Group>(null);

  const indexMCPRef = useRef<Group>(null);
  const indexPIPRef = useRef<Group>(null);
  const indexDIPRef = useRef<Group>(null);

  const middleMCPRef = useRef<Group>(null);
  const middlePIPRef = useRef<Group>(null);
  const middleDIPRef = useRef<Group>(null);

  const ringMCPRef = useRef<Group>(null);
  const ringPIPRef = useRef<Group>(null);
  const ringDIPRef = useRef<Group>(null);

  const pinkyMCPRef = useRef<Group>(null);
  const pinkyPIPRef = useRef<Group>(null);
  const pinkyDIPRef = useRef<Group>(null);

  useFrame(() => {
    const { normalized, derived } = data;
    const D = normalized;
    const alpha = 0.25; 
    
    const clamp = (val: number, min: number, max: number) => Math.max(min, Math.min(max, val));
    
    // --- THUMB CMC ---
    if (thumbCMCRef.current && thumbMCPRef.current && thumbIPRef.current) {
      
      // 根据最新要求：
      // 1号传感器 (THUMB_IP, index 0) 对应指间关节 (IP)
      // 2号传感器 (THUMB_MCP, index 1) 对应腕掌关节内扣程度 (CMC)
      // 掌指关节 (MCP) 为两者的加权计算值
      const ipVal = clamp(D[SENSOR_MAP.THUMB_IP], 0, 1);
      const cmcTuck = clamp(D[SENSOR_MAP.THUMB_MCP], 0, 1);
      const mcpVal = (ipVal * 0.4) + (cmcTuck * 0.6);
      const abduction = clamp(derived.thumbAbduction ?? 0, 0, 1);

      // === 拇指腕掌关节 (CMC) — 鞍状关节 3-DOF 四元数模型 ===
      // 参考文献:
      //   - Hollister 1992 (J Orthop Res): FE 轴在大多角骨, AA 轴在第1掌骨, 两轴不垂直不相交
      //   - Halilaj 2013 (J Biomech): Ztpm-Y-Xmc1 Euler 序列, 基于鞍状关节面几何
      //   - Chang 2008 (TBME): 3 旋转轴, FE/AA 主导, PS 对掌时可达 23°
      //   - Symes 2025 (Front Robot AI "Anthro-Thumb"): opposition 是独立控制通道
      //
      // 实现方式: 用四元数 (Quaternion) 组合,避免 Euler 角的万向锁和顺序依赖问题
      // 每个 DOF 绕其真正的生物力学轴旋转, 然后用四元数乘法组合
      // 组合顺序 (右到左应用): PS(骨自身扭转) → AA(掌平面收展) → FE(屈伸)

      // 基础静止姿态四元数 (基准偏转,让拇指自然位于掌侧)
      const baseAngleX = 0.1;   // 微抬
      const baseAngleY = -0.05; // 初始扭转,指腹朝掌心
      const baseAngleZ = -0.25; // 基准偏深

      // --- DOF 1: 屈伸 (FE) — 绕 X 轴 ---
      // cmcTuck 高 → 屈曲(弯向掌心); abduction 高 → 伸展(抬起远离掌心)
      const angleFE = baseAngleX + cmcTuck * 1.6 - abduction * 1.8;

      // --- DOF 2: 收展 (AA) — 绕 Z 轴 ---
      // cmcTuck 高 → 内收(对掌方向,扫向小指); abduction 高 → 外展(掌平面内打开)
      const angleAA = baseAngleZ + cmcTuck * 1.7 - abduction * 1.0;

      // --- DOF 3: 旋前/旋后 (PS) — 绕 Y 轴 (骨长轴) ---
      // cmcTuck 高 → 旋前(指腹转向掌心, 对掌核心); abduction 高 → 旋后(指腹朝外)
      const anglePS = baseAngleY + cmcTuck * 1.8 + abduction * 1.4;

      // 构建各 DOF 的独立四元数
      const qFE = new Quaternion().setFromAxisAngle(AXIS_X, angleFE);
      const qAA = new Quaternion().setFromAxisAngle(AXIS_Z, angleAA);
      const qPS = new Quaternion().setFromAxisAngle(AXIS_Y, anglePS);

      // 组合: qTotal = qFE * qAA * qPS (右乘=先应用右边的)
      // 几何意义: 先旋前扭转骨自身 → 再在掌平面内收展 → 最后屈伸
      const qTarget = new Quaternion().multiplyQuaternions(qFE, qAA).multiply(qPS);

      // MCP (掌指关节)
      let targetMcpZ = 0.0;
      let targetMcpX = -mcpVal * 1.2;

      // IP (指间关节)
      let targetIpZ = 0.0;
      let targetIpX = -ipVal * 1.4;

      // 如果有 thumbTuck 标定数据，使用它吸附到贴掌心的极限位置，防止穿模
      if (calibration?.isCalibrated && calibration.thumbTuck) {
        const tp = calibration.thumbTuck;
        const currentRaw = data.raw;
        const tIdx = [SENSOR_MAP.THUMB_IP, SENSOR_MAP.THUMB_MCP, SENSOR_MAP.THUMB_SPREAD];

        const dist = Math.sqrt(tIdx.reduce((acc, i) => acc + Math.pow(currentRaw[i] - (tp[i] || 0), 2), 0));

        if (dist < 400) {
          const snap = 1.0 - (dist / 400);
          // 贴紧掌心时的目标四元数 (指腹贴掌心: 强力屈曲+内收+旋前)
          const qTuck = new Quaternion()
            .multiply(new Quaternion().setFromAxisAngle(AXIS_X, -0.2))
            .multiply(new Quaternion().setFromAxisAngle(AXIS_Y, 0.5))
            .multiply(new Quaternion().setFromAxisAngle(AXIS_Z, 1.5));
          qTarget.slerp(qTuck, snap);

          targetMcpZ = MathUtils.lerp(targetMcpZ, 0.0, snap);
          targetMcpX = MathUtils.lerp(targetMcpX, -0.4, snap);
          targetIpZ = MathUtils.lerp(targetIpZ, 0.0, snap);
          targetIpX = MathUtils.lerp(targetIpX, -0.2, snap);
        }
      }

      // 用 slerp (球面线性插值) 平滑过渡四元数,避免 Euler 角插值的万向锁问题
      thumbCMCRef.current.quaternion.slerp(qTarget, alpha);

      // 应用指间关节的旋转，主要在Z轴体现收拢，副在-X轴体现向内贴紧
      thumbMCPRef.current.rotation.z = MathUtils.lerp(thumbMCPRef.current.rotation.z, targetMcpZ, alpha);
      thumbMCPRef.current.rotation.x = MathUtils.lerp(thumbMCPRef.current.rotation.x, targetMcpX, alpha);
      
      thumbIPRef.current.rotation.z = MathUtils.lerp(thumbIPRef.current.rotation.z, targetIpZ, alpha);
      thumbIPRef.current.rotation.x = MathUtils.lerp(thumbIPRef.current.rotation.x, targetIpX, alpha);
    }

    // --- FINGERS HELPER ---
    const flexFinger = (
      mcpRef: React.RefObject<Group>, 
      pipRef: React.RefObject<Group>, 
      dipRef: React.RefObject<Group>, 
      mcpVal: number, 
      pipVal: number, 
      dipVal: number,
      abdVal: number = 0,
      maxAbdFactor: number = 1.0
    ) => {
        if (mcpRef.current) {
            const flex = Math.min(mcpVal * 2.2, 1.6); 
            mcpRef.current.rotation.x = MathUtils.lerp(mcpRef.current.rotation.x, flex, alpha);
            
            // 增加张开的指数曲线减轻手指弯曲时其他数据波动带来的干扰，并放大最终显示的张开效果 (特别增强无小指张开效果)
            const sign = abdVal < 0 ? -1 : 1;
            const curvedAbd = sign * Math.pow(Math.abs(abdVal), 1.5);
            const abd = clamp(curvedAbd * 3.5, -0.6 * maxAbdFactor, 1.2 * maxAbdFactor) * 0.4 * mirror;
            mcpRef.current.rotation.z = MathUtils.lerp(mcpRef.current.rotation.z, -abd, alpha);
        }
        if (pipRef.current) {
            const flex = Math.min(pipVal * 2.2, 1.8);
            pipRef.current.rotation.x = MathUtils.lerp(pipRef.current.rotation.x, flex, alpha);
        }
        if (dipRef.current) {
            const flex = Math.min(dipVal * 2.2, 1.5);
            dipRef.current.rotation.x = MathUtils.lerp(dipRef.current.rotation.x, flex, alpha);
        }
    };

    flexFinger(indexMCPRef, indexPIPRef, indexDIPRef, derived.indexWeightedMCP, D[SENSOR_MAP.INDEX_PIP], derived.indexDIP, derived.indexAbduction, 1.4);
    flexFinger(middleMCPRef, middlePIPRef, middleDIPRef, derived.middleWeightedMCP, D[SENSOR_MAP.MIDDLE_PIP], derived.middleDIP, derived.middleAbduction, 1.0);
    flexFinger(ringMCPRef, ringPIPRef, ringDIPRef, D[SENSOR_MAP.RING_PIP], D[SENSOR_MAP.RING_PIP], derived.ringDIP, derived.ringAbduction, 1.0); 
    flexFinger(pinkyMCPRef, pinkyPIPRef, pinkyDIPRef, D[SENSOR_MAP.PINKY_PIP], D[SENSOR_MAP.PINKY_PIP], derived.pinkyDIP, derived.pinkyAbduction, 2.0);
  });

  return (
    <group ref={groupRef} position={position} scale={[1.2 * mirror, 1.2, 1.2]}>
        {/* PALM BODY */}
        <mesh position={[0, 1.3, -0.15]} castShadow receiveShadow>
            <boxGeometry args={[2.3, 2.6, 0.55]} />
            <meshStandardMaterial color={boneColor} roughness={0.7} />
        </mesh>

        {/* THUMB - Wrapped group sets the local axes to accurately simulate human thumb biomechanics */}
        <group position={[1.2, 0.25, 0.25]} rotation={[0, 0, -0.5]}>
            <group ref={thumbCMCRef}>
                <Joint radius={0.48} color={jointColor} />
                <Bone radius={0.38} length={1.25} color={boneColor} />
                <group position={[0, 1.25, 0]} ref={thumbMCPRef}>
                    <Joint radius={0.42} color={jointColor} />
                    <Bone radius={0.35} length={1.0} color={boneColor} />
                    <group position={[0, 1.0, 0]} ref={thumbIPRef}>
                        <Joint radius={0.36} color={jointColor} />
                        <Bone radius={0.32} length={0.8} color={boneColor} />
                    </group>
                </group>
            </group>
        </group>

        {/* INDEX */}
        <group position={[0.75, 2.6, 0]} ref={indexMCPRef}>
             <Joint radius={0.38} color={jointColor} />
             <Bone radius={0.3} length={1.1} color={boneColor} />
             <group position={[0, 1.1, 0]} ref={indexPIPRef}>
                 <Joint radius={0.34} color={jointColor} />
                 <Bone radius={0.27} length={0.9} color={boneColor} />
                 <group position={[0, 0.9, 0]} ref={indexDIPRef}>
                     <Joint radius={0.3} color={jointColor} />
                     <Bone radius={0.24} length={0.7} color={boneColor} />
                 </group>
             </group>
        </group>

        {/* MIDDLE */}
        <group position={[0.2, 2.6, 0]} ref={middleMCPRef}>
             <Joint radius={0.4} color={jointColor} />
             <Bone radius={0.32} length={1.2} color={boneColor} />
             <group position={[0, 1.2, 0]} ref={middlePIPRef}>
                 <Joint radius={0.35} color={jointColor} />
                 <Bone radius={0.28} length={1.0} color={boneColor} />
                 <group position={[0, 1.0, 0]} ref={middleDIPRef}>
                     <Joint radius={0.32} color={jointColor} />
                     <Bone radius={0.25} length={0.8} color={boneColor} />
                 </group>
             </group>
        </group>

        {/* RING */}
        <group position={[-0.35, 2.6, 0]} ref={ringMCPRef}>
             <Joint radius={0.38} color={jointColor} />
             <Bone radius={0.3} length={1.15} color={boneColor} />
             <group position={[0, 1.15, 0]} ref={ringPIPRef}>
                 <Joint radius={0.34} color={jointColor} />
                 <Bone radius={0.27} length={0.9} color={boneColor} />
                 <group position={[0, 0.9, 0]} ref={ringDIPRef}>
                     <Joint radius={0.3} color={jointColor} />
                     <Bone radius={0.24} length={0.75} color={boneColor} />
                 </group>
             </group>
        </group>

        {/* PINKY */}
        <group position={[-0.85, 2.5, 0]} ref={pinkyMCPRef}>
             <Joint radius={0.35} color={jointColor} />
             <Bone radius={0.26} length={0.9} color={boneColor} />
             <group position={[0, 0.9, 0]} ref={pinkyPIPRef}>
                 <Joint radius={0.3} color={jointColor} />
                 <Bone radius={0.24} length={0.7} color={boneColor} />
                 <group position={[0, 0.7, 0]} ref={pinkyDIPRef}>
                     <Joint radius={0.28} color={jointColor} />
                     <Bone radius={0.2} length={0.6} color={boneColor} />
                 </group>
             </group>
        </group>
    </group>
  );
};
