import { CreepModelParams, CreepFilterState } from '../types';

const DEFAULT_PARAMS: CreepModelParams = {
  tauLoad: 0.2,
  tauUnload: 0.4,
  alphaLoad: 0.08,
  alphaUnload: 0.06,
  hysteresisGap: 0.02,
};

const DEFAULT_STATE: CreepFilterState = {
  creepOffset: 0,
  prevValue: 0,
  prevCompensated: 0,
  epsilonLoad: 0,
  epsilonUnload: 0,
  prevDirection: 0,
};

/**
 * 蠕变补偿滤波器（方向感知双参数模型）
 *
 * 液金/TPU 传感器的粘弹性蠕变在加载和卸载方向表现不同：
 *   加载（弯折）：V(t) = ΔV · (1 - exp(-t/τ_load))
 *   卸载（松开）：V(t) = ΔV · exp(-t/τ_unload)
 *
 * 本滤波器维护两个独立的蠕变状态量 epsilonLoad / epsilonUnload，
 * 根据每帧运动方向分别使用对应的 tau 和 alpha 进行更新。
 */
export class CreepFilter {
  private params: CreepModelParams[];
  private states: CreepFilterState[];
  private sampleRate: number;
  private dt: number;

  constructor(sampleRate: number = 40) {
    this.sampleRate = sampleRate;
    this.dt = 1.0 / sampleRate;
    this.params = Array(12).fill(null).map(() => ({ ...DEFAULT_PARAMS }));
    this.states = Array(12).fill(null).map(() => ({ ...DEFAULT_STATE }));
  }

  setParams(channel: number, params: Partial<CreepModelParams>) {
    if (channel >= 0 && channel < 12) {
      this.params[channel] = { ...this.params[channel], ...params };
    }
  }

  setAllParams(params: CreepModelParams[]) {
    params.forEach((p, i) => {
      if (i < 12) {
        this.params[i] = { ...this.params[i], ...p };
      }
    });
  }

  reset() {
    this.states = Array(12).fill(null).map(() => ({ ...DEFAULT_STATE }));
  }

  resetChannel(channel: number) {
    if (channel >= 0 && channel < 12) {
      this.states[channel] = { ...DEFAULT_STATE };
    }
  }

  process(rawValues: number[]): number[] {
    const compensated = new Array(12).fill(0);

    for (let i = 0; i < 12; i++) {
      const raw = rawValues[i] || 0;
      const params = this.params[i];
      const state = this.states[i];

      const delta = raw - state.prevValue;
      const direction = delta > 0 ? 1 : delta < 0 ? -1 : state.prevDirection;

      // 滞回补偿：比例模型
      const h = Math.sign(delta) * params.hysteresisGap * Math.abs(delta);
      const y = raw - h;

      // 根据方向选择对应的 tau 和 alpha
      const isLoading = direction > 0;
      const tau = isLoading ? params.tauLoad : params.tauUnload;
      const alpha = isLoading ? params.alphaLoad : params.alphaUnload;

      // 更新对应的蠕变状态量
      const epsilon = isLoading ? state.epsilonLoad : state.epsilonUnload;

      // 蠕变状态量的微分方程：dε/dt = -ε/τ + α·(dV/dt)
      const depsilon_dt = -(epsilon / tau) + alpha * (delta / this.dt);
      const newEpsilon = epsilon + depsilon_dt * this.dt;

      // 对侧状态量自然衰减（交叉项，防止方向切换时残余误差）
      const oppositeEpsilon = isLoading ? state.epsilonUnload : state.epsilonLoad;
      const oppositeTau = isLoading ? params.tauUnload : params.tauLoad;
      const newOppositeEpsilon = oppositeEpsilon * Math.exp(-this.dt / oppositeTau);

      // 总蠕变量为两个方向状态量之和
      const totalEpsilon = newEpsilon + newOppositeEpsilon;
      const compensatedValue = y - totalEpsilon;

      // 更新状态
      state.creepOffset = totalEpsilon;
      state.prevValue = raw;
      state.prevCompensated = compensatedValue;
      if (isLoading) {
        state.epsilonLoad = newEpsilon;
        state.epsilonUnload = newOppositeEpsilon;
      } else {
        state.epsilonUnload = newEpsilon;
        state.epsilonLoad = newOppositeEpsilon;
      }
      state.prevDirection = direction;

      compensated[i] = compensatedValue;
    }

    return compensated;
  }

  getState(channel: number): CreepFilterState | null {
    if (channel >= 0 && channel < 12) {
      return { ...this.states[channel] };
    }
    return null;
  }

  getParams(channel: number): CreepModelParams | null {
    if (channel >= 0 && channel < 12) {
      return { ...this.params[channel] };
    }
    return null;
  }

  getAllParams(): CreepModelParams[] {
    return this.params.map(p => ({ ...p }));
  }
}

export const creepFilter = new CreepFilter(40);
