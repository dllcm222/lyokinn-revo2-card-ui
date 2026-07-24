
export type HandSide = 'LEFT' | 'RIGHT';

export interface SensorReadings {
  raw: number[];        // The 12 raw integer values from serial
  normalized: number[]; // The 12 mapped values (0.0 - 1.0)
  derived: {            // Calculated additional degrees of freedom
    thumbOpposition: number;
    indexDIP: number;
    middleDIP: number;
    ringDIP: number;
    pinkyDIP: number;
    indexAbduction: number;
    middleAbduction: number;
    ringAbduction: number;
    pinkyAbduction: number;
    indexWeightedMCP: number;
    middleWeightedMCP: number;
  };
}

export interface CalibrationRange {
  min: number;
  max: number;
  exponent: number;
}

export interface HandCalibration {
  ranges: CalibrationRange[];
  thumbTuck?: number[] | null;
  isCalibrated: boolean;
}

export interface CreepModelParams {
  /** 加载（弯折）蠕变时间常数 */
  tauLoad: number;
  /** 卸载（松开）蠕变恢复时间常数 */
  tauUnload: number;
  /** 加载蠕变幅值系数 */
  alphaLoad: number;
  /** 卸载蠕变恢复幅值系数 */
  alphaUnload: number;
  /** 滞回间隙 */
  hysteresisGap: number;
}

export interface CreepFilterState {
  creepOffset: number;
  prevValue: number;
  prevCompensated: number;
  /** 加载方向累积的蠕变状态量 */
  epsilonLoad: number;
  /** 卸载方向累积的蠕变状态量 */
  epsilonUnload: number;
  /** 上一帧的运动方向：1=加载, -1=卸载, 0=静止 */
  prevDirection: number;
}

export interface CARDProcessingResult {
  raw: number[];
  compensated: number[];
  decoupled: number[];
  filtered: number[];
  normalized: number[];
  derived: SensorReadings['derived'];
}

export interface MLPLayer {
  weights: number[][];
  biases: number[];
}

export interface MLPWeights {
  input: MLPLayer;
  hidden: MLPLayer;
  output: MLPLayer;
}

export interface DecouplerState {
  previousOutput: number[];
}

export interface CreepCalibrationResult {
  tauLoad: number;
  tauUnload: number;
  alphaLoad: number;
  alphaUnload: number;
  loadCurvePoints: { time: number; value: number }[];
  unloadCurvePoints: { time: number; value: number }[];
  rSquaredLoad: number;
  rSquaredUnload: number;
}

export interface CreepFitResult extends CreepModelParams {
  rSquaredLoad: number;
  rSquaredUnload: number;
}

export enum CalibrationStep {
  IDLE = 'IDLE',
  RELAX = 'RELAX',
  FIST = 'FIST',
  FLAT = 'FLAT',
  SPREAD = 'SPREAD',
  THUMB_TUCK = 'THUMB_TUCK',
  CREEP = 'CREEP',
  DECOUPLER = 'DECOUPLER',
}

// Precise mapping from image provided (1-12 sequence):
export const SENSOR_MAP = {
  THUMB_IP: 0,            // #1
  THUMB_MCP: 1,           // #2
  THUMB_SPREAD: 2,        // #3
  INDEX_PIP: 3,           // #4
  INDEX_MIDDLE_SPREAD: 4, // #5
  INDEX_MCP: 5,           // #6
  MIDDLE_PIP: 6,          // #7
  MIDDLE_RING_SPREAD: 7,  // #8
  MIDDLE_MCP: 8,          // #9 
  RING_PIP: 9,            // #10
  RING_PINKY_SPREAD: 10,  // #11
  PINKY_PIP: 11,          // #12
};

export const HW_MAP = {
  T_IP: 0,   // #1
  T_MCP: 1,  // #2
  T_SPD: 2,  // #3
  I_IP: 3,   // #4
  IM_SPD: 4, // #5
  I_MCP: 5,  // #6
  M_IP: 6,   // #7
  MR_SPD: 7, // #8
  M_MCP: 8,  // #9
  R_IP: 9,   // #10
  RP_SPD: 10, // #11
  P_IP: 11,  // #12
};
