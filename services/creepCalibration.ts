import { CreepCalibrationResult, CreepModelParams, CreepFitResult } from '../types';

const DEFAULT_PARAMS: CreepModelParams = {
  tauLoad: 0.2,
  tauUnload: 0.4,
  alphaLoad: 0.08,
  alphaUnload: 0.06,
  hysteresisGap: 0.02,
};

/**
 * 蠕变标定器
 *
 * 支持非对称加载/卸载蠕变模型：
 *   加载:  V_load(t)  = ΔV · (1 - exp(-t/τ_load))  + V_start
 *   卸载:  V_unload(t) = ΔV · exp(-t/τ_unload)      + V_rest
 *
 * 液金/TPU 复合膜是粘弹性材料，加载和卸载的时间常数通常不同，
 * 卸载恢复 τ_unload 一般大于加载 τ_load。
 */
export class CreepCalibrator {
  private buffer: { time: number; value: number }[] = [];
  private startTime: number | null = null;
  private isRecording: boolean = false;

  startRecording() {
    this.buffer = [];
    this.startTime = Date.now();
    this.isRecording = true;
  }

  stopRecording() {
    this.isRecording = false;
    this.startTime = null;
  }

  isRecordingState(): boolean {
    return this.isRecording;
  }

  addSample(value: number) {
    if (!this.isRecording || this.startTime === null) return;
    const time = (Date.now() - this.startTime) / 1000;
    this.buffer.push({ time, value });
  }

  getBuffer(): { time: number; value: number }[] {
    return [...this.buffer];
  }

  clearBuffer() {
    this.buffer = [];
  }

  /**
   * 指数蠕变模型拟合（通用），支持加载/卸载方向
   *
   * 加载模型:  V(t) = ΔV · (1 - exp(-t/τ)) + V_start   （单调趋向 V_end）
   * 卸载模型:  V(t) = ΔV · exp(-t/τ) + V_rest           （单调趋向 V_rest）
   *
   * 通过判断 deltaTotal 符号自动选择模型方向。
   */
  fitExponentialCreep(samples: { time: number; value: number }[]): {
    tau: number;
    alpha: number;
    rSquared: number;
  } {
    if (samples.length < 30) {
      return { tau: 0.2, alpha: 0.01, rSquared: 0 };
    }

    const n = samples.length;
    const vStart = samples.slice(0, 5).reduce((s, p) => s + p.value, 0) / 5;
    const vEnd = samples.slice(-30).reduce((s, p) => s + p.value, 0) / 30;
    const deltaTotal = vEnd - vStart;

    if (Math.abs(deltaTotal) < 5) {
      return { tau: 0.5, alpha: 0.01, rSquared: 0 };
    }

    const isLoad = deltaTotal > 0;

    // 网格搜索最优 tau
    const tauCandidates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0];
    let bestTau = 0.2;
    let bestSSRes = Infinity;

    for (const tau of tauCandidates) {
      let ssRes = 0;
      for (const p of samples) {
        const predicted = isLoad
          ? deltaTotal * (1 - Math.exp(-p.time / tau)) + vStart
          : deltaTotal * Math.exp(-p.time / tau) + vEnd;
        ssRes += Math.pow(p.value - predicted, 2);
      }
      if (ssRes < bestSSRes) {
        bestSSRes = ssRes;
        bestTau = tau;
      }
    }

    // 在最优 tau 附近细化搜索
    const fineStep = bestTau * 0.1;
    for (let t = bestTau - fineStep * 5; t <= bestTau + fineStep * 5; t += fineStep) {
      if (t <= 0) continue;
      let ssRes = 0;
      for (const p of samples) {
        const predicted = isLoad
          ? deltaTotal * (1 - Math.exp(-p.time / t)) + vStart
          : deltaTotal * Math.exp(-p.time / t) + vEnd;
        ssRes += Math.pow(p.value - predicted, 2);
      }
      if (ssRes < bestSSRes) {
        bestSSRes = ssRes;
        bestTau = t;
      }
    }

    // alpha: 蠕变幅值相对于 ADC 量程的比例，按采样率缩放
    const timeSpan = samples[n - 1].time - samples[0].time || 1;
    const sampleRate = n / timeSpan;
    const alpha = (Math.abs(deltaTotal) / 4095) * (sampleRate * 0.1);

    // 计算 R²
    const meanY = samples.reduce((s, p) => s + p.value, 0) / n;
    let ssTot = 0;
    for (const p of samples) {
      ssTot += Math.pow(p.value - meanY, 2);
    }
    const rSquared = ssTot > 0 ? 1 - bestSSRes / ssTot : 0;

    return {
      tau: Math.max(0.01, Math.min(5.0, bestTau)),
      alpha: Math.max(0.001, Math.min(0.5, alpha)),
      rSquared,
    };
  }

  /**
   * 从标定缓冲区批量拟合所有 12 个通道的加载/卸载蠕变参数
   *
   * @param loadBuffer   加载阶段数据（握拳保持）
   * @param unloadBuffer 卸载阶段数据（松手保持）
   * @param sampleRate   采样率 (Hz)
   */
  calibrateAllChannelsFromBuffer(
    loadBuffer: number[][],
    unloadBuffer: number[][],
    sampleRate: number = 40
  ): CreepFitResult[] {
    const results: CreepFitResult[] = [];

    for (let channel = 0; channel < 12; channel++) {
      const loadSamples = loadBuffer.map((data, index) => ({
        time: index / sampleRate,
        value: data[channel] || 0,
      }));
      const unloadSamples = unloadBuffer.map((data, index) => ({
        time: index / sampleRate,
        value: data[channel] || 0,
      }));

      const loadFit = this.fitExponentialCreep(loadSamples);
      const unloadFit = this.fitExponentialCreep(unloadSamples);

      // 卸载恢复时间常数通常应大于加载，若拟合值偏小则取加载值的 1.5 倍作为下限
      const tauUnload = Math.max(unloadFit.tau, loadFit.tau * 1.5);

      results.push({
        tauLoad: loadFit.tau,
        tauUnload,
        alphaLoad: loadFit.alpha,
        alphaUnload: unloadFit.alpha,
        hysteresisGap: 0.02,
        rSquaredLoad: loadFit.rSquared,
        rSquaredUnload: unloadFit.rSquared,
      });
    }

    return results;
  }

  /**
   * 从快速屈伸循环数据估算滞回间隙（多周期平均）
   *
   * 改进点：
   * 1. 提取所有上升沿/下降沿过中点对，而非仅第一个
   * 2. 对所有有效周期的 lagRatio 取中位数，抑制异常值
   */
  estimateHysteresisGap(cycleData: number[][]): number {
    const gaps: number[] = [];

    for (let ch = 0; ch < 12; ch++) {
      const values = cycleData.map(d => d[ch] || 0);
      if (values.length < 10) continue;

      const max = Math.max(...values);
      const min = Math.min(...values);
      const range = max - min;
      if (range < 10) continue;

      const mid = (max + min) / 2;

      // 收集所有上升沿和下降沿过中点位置
      const risingIndices: number[] = [];
      const fallingIndices: number[] = [];

      for (let i = 1; i < values.length; i++) {
        if (values[i - 1] < mid && values[i] >= mid) {
          risingIndices.push(i);
        } else if (values[i - 1] >= mid && values[i] < mid) {
          fallingIndices.push(i);
        }
      }

      // 配对：每个上升沿找其后的第一个下降沿
      let pairCount = 0;
      let lagSum = 0;
      for (const rIdx of risingIndices) {
        const fIdx = fallingIndices.find(f => f > rIdx);
        if (fIdx !== undefined) {
          const lagRatio = (fIdx - rIdx) / values.length;
          lagSum += lagRatio;
          pairCount++;
        }
      }

      if (pairCount > 0) {
        gaps.push((lagSum / pairCount) * 0.15);
      }
    }

    if (gaps.length === 0) return 0.02;

    // 取中位数而非均值，抑制异常值
    gaps.sort((a, b) => a - b);
    const mid = Math.floor(gaps.length / 2);
    const median = gaps.length % 2 === 0 ? (gaps[mid - 1] + gaps[mid]) / 2 : gaps[mid];

    return Math.max(0.01, Math.min(0.1, median));
  }

  /**
   * 从瞬态响应数据粗估蠕变参数（降级方案，rSquared 不足时使用）
   */
  estimateFromTransient(data: number[], isLoad: boolean = true): Partial<CreepModelParams> {
    if (data.length < 5) {
      return isLoad
        ? { tauLoad: 0.2, alphaLoad: 0.08 }
        : { tauUnload: 0.4, alphaUnload: 0.06 };
    }

    const diffs: number[] = [];
    for (let i = 1; i < data.length; i++) {
      diffs.push(Math.abs(data[i] - data[i - 1]));
    }

    const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    const maxDiff = Math.max(...diffs);

    const tau = Math.max(0.05, Math.min(0.5, (avgDiff / maxDiff) * 0.3));
    const alpha = Math.max(0.01, Math.min(0.2, avgDiff / 4095));

    return isLoad
      ? { tauLoad: tau, alphaLoad: alpha }
      : { tauUnload: Math.max(tau, 0.3), alphaUnload: alpha };
  }

  /**
   * 兼容旧接口：单缓冲区拟合（仅加载方向）
   * @deprecated 请使用 calibrateAllChannelsFromBuffer 传入 load/unload 双缓冲区
   */
  calibrateAllChannels(rawDataHistory: number[][]): CreepModelParams[] {
    const results: CreepModelParams[] = [];

    for (let channel = 0; channel < 12; channel++) {
      this.buffer = [];
      rawDataHistory.forEach((data, index) => {
        const time = index / 40;
        this.buffer.push({ time, value: data[channel] || 0 });
      });

      const fit = this.fitExponentialCreep(this.buffer);
      results.push({
        tauLoad: fit.tau,
        tauUnload: fit.tau * 1.5,
        alphaLoad: fit.alpha,
        alphaUnload: fit.alpha * 0.8,
        hysteresisGap: 0.02,
      });
    }

    return results;
  }

  /**
   * 兼容旧接口：线性回归拟合（不推荐）
   * @deprecated 请使用 fitExponentialCreep
   */
  fitCreepCurve(): CreepCalibrationResult {
    if (this.buffer.length < 10) {
      return {
        tauLoad: 0.2,
        tauUnload: 0.4,
        alphaLoad: 0.08,
        alphaUnload: 0.06,
        loadCurvePoints: [],
        unloadCurvePoints: [],
        rSquaredLoad: 0,
        rSquaredUnload: 0,
      };
    }

    const points = this.buffer;
    const n = points.length;

    let sumT = 0, sumY = 0, sumTY = 0, sumT2 = 0;
    for (const p of points) {
      sumT += p.time;
      sumY += p.value;
      sumTY += p.time * p.value;
      sumT2 += p.time * p.time;
    }

    const denom = n * sumT2 - sumT * sumT;
    if (Math.abs(denom) < 1e-10) {
      return {
        tauLoad: 0.2,
        tauUnload: 0.4,
        alphaLoad: 0.08,
        alphaUnload: 0.06,
        loadCurvePoints: points,
        unloadCurvePoints: [],
        rSquaredLoad: 0,
        rSquaredUnload: 0,
      };
    }

    const slope = (n * sumTY - sumT * sumY) / denom;
    const intercept = (sumY - slope * sumT) / n;

    let ssTot = 0, ssRes = 0;
    for (const p of points) {
      const pred = slope * p.time + intercept;
      ssTot += Math.pow(p.value - sumY / n, 2);
      ssRes += Math.pow(p.value - pred, 2);
    }

    const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0;

    const tau = slope > 0 ? 1.0 / slope : 0.2;
    const alpha = Math.abs(intercept) / 4095;

    return {
      tauLoad: Math.max(0.01, Math.min(1.0, tau)),
      tauUnload: Math.max(0.01, Math.min(1.0, tau * 1.5)),
      alphaLoad: Math.max(0.001, Math.min(0.5, alpha)),
      alphaUnload: Math.max(0.001, Math.min(0.5, alpha * 0.8)),
      loadCurvePoints: points,
      unloadCurvePoints: [],
      rSquaredLoad: rSquared,
      rSquaredUnload: 0,
    };
  }
}

export const creepCalibrator = new CreepCalibrator();

export { DEFAULT_PARAMS };
