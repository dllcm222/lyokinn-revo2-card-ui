import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Brain,
  RefreshCw,
  Download,
  ArrowLeft,
  CheckCircle,
  Circle,
  Loader,
  AlertTriangle,
} from 'lucide-react';
import {
  MLPTrainer,
  FINGER_CHANNELS,
  TRAINING_STEPS,
  TrainingStep,
  TrainingProgress,
  prepareTrainingData,
} from './services/mlpTrainer';
import { mlpDecoupler } from './services/mlpDecoupler';
import { MLPWeights } from './types';

const SAMPLE_RATE = 40;
const REST_DURATION = 20; // frames for baseline
const RECORD_DURATION = 400; // 10 seconds @ 40Hz
const TARGET_FRAMES = RECORD_DURATION;

interface Props {
  onBack: () => void;
  onWeightsApplied: (baseline: number[], maxDelta: number[]) => void;
  connected: boolean;
  rawData: number[];
}

export default function TrainingMode({ onBack, onWeightsApplied, connected, rawData }: Props) {
  const [currentStep, setCurrentStep] = useState<number>(0); // 0=rest, 1-5=fingers
  const [isRecording, setIsRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [collectedSegments, setCollectedSegments] = useState<Record<string, number[][]>>({});
  const [baseline, setBaseline] = useState<number[]>([]);
  const [maxDelta, setMaxDelta] = useState<number[]>([]);
  const [phase, setPhase] = useState<'collect' | 'train' | 'done'>('collect');

  // Training state
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [trainedWeights, setTrainedWeights] = useState<MLPWeights | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  // Training config
  const [epochs, setEpochs] = useState(300);
  const [learningRate, setLearningRate] = useState(0.001);

  const recordingRef = useRef<number[][]>([]);
  const rawDataRef = useRef<number[]>(rawData);

  useEffect(() => {
    rawDataRef.current = rawData;
  }, [rawData]);

  // Recording tick
  useEffect(() => {
    if (!isRecording) return;
    const interval = setInterval(() => {
      recordingRef.current.push([...rawDataRef.current]);
      setFrameCount(prev => prev + 1);
    }, 1000 / SAMPLE_RATE);
    return () => clearInterval(interval);
  }, [isRecording]);

  const stepInfo = TRAINING_STEPS[currentStep];
  const isRest = stepInfo === 'rest';
  const fingerInfo = !isRest ? FINGER_CHANNELS[stepInfo] : null;
  const totalSteps = TRAINING_STEPS.length;
  const completedSteps = Object.keys(collectedSegments).length;

  const startRecording = useCallback(() => {
    recordingRef.current = [];
    setFrameCount(0);
    setIsRecording(true);
  }, []);

  const stopRecording = useCallback(() => {
    setIsRecording(false);
    const frames = recordingRef.current;

    if (isRest) {
      // Compute baseline from rest frames
      if (frames.length >= 10) {
        const avg = new Array(12).fill(0);
        const count = Math.min(frames.length, REST_DURATION);
        for (let i = 0; i < count; i++) {
          for (let ch = 0; ch < 12; ch++) {
            avg[ch] += frames[i][ch];
          }
        }
        setBaseline(avg.map(v => v / count));
      }
      setCollectedSegments(prev => ({ ...prev, rest: frames }));
    } else {
      setCollectedSegments(prev => ({ ...prev, [stepInfo]: frames }));
    }

    // Auto advance
    if (currentStep < totalSteps - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      setPhase('train');
    }
  }, [currentStep, isRest, stepInfo, totalSteps]);

  // Auto-stop when enough frames
  useEffect(() => {
    if (isRecording && frameCount >= TARGET_FRAMES) {
      stopRecording();
    }
  }, [frameCount, isRecording, stopRecording]);

  const handleTrain = async () => {
    if (!baseline.length) {
      setTrainingError('缺少静止基线数据，请返回重新采集');
      return;
    }

    setTraining(true);
    setTrainingError(null);
    setTrainingProgress(null);

    try {
      const { inputs, targets, maxDelta: calcMaxDelta } = prepareTrainingData(collectedSegments, baseline);
      setMaxDelta(calcMaxDelta);

      if (inputs.length < 50) {
        setTrainingError('训练数据不足，请确保每个动作至少采集了 50 帧');
        setTraining(false);
        return;
      }

      const trainer = new MLPTrainer();
      await trainer.train(inputs, targets, {
        epochs,
        learningRate,
        printInterval: Math.max(1, Math.floor(epochs / 20)),
      }, (progress) => {
        setTrainingProgress(progress);
      });

      const weights = trainer.exportWeights();
      setTrainedWeights(weights);
      setPhase('done');

      // Save maxDelta alongside weights for inference normalization
      const fullResult = { weights, baseline, maxDelta };
      localStorage.setItem('mlpWeights', JSON.stringify(fullResult));
    } catch (e: any) {
      setTrainingError(`训练失败: ${e.message}`);
    } finally {
      setTraining(false);
    }
  };

  const applyWeights = () => {
    if (!trainedWeights) return;
    mlpDecoupler.setWeights(trainedWeights);
    mlpDecoupler.reset();
    mlpDecoupler.setEnabled(true);
    onWeightsApplied(baseline, maxDelta);
  };

  const exportData = () => {
    const data = JSON.stringify({ baseline, segments: collectedSegments }, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mlp-training-data-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportWeights = () => {
    if (!trainedWeights) return;
    const data = JSON.stringify({
      weights: trainedWeights,
      baseline,
      maxDelta,
    }, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mlp-weights-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetCollection = () => {
    setCollectedSegments({});
    setBaseline([]);
    setCurrentStep(0);
    setPhase('collect');
    setTrainedWeights(null);
    setTrainingProgress(null);
    setTrainingError(null);
  };

  // ─── Render ───
  return (
    <div className="h-full flex flex-col bg-gray-950 text-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="p-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors">
            <ArrowLeft size={16} />
          </button>
          <div>
            <h1 className="text-sm font-bold text-purple-300 flex items-center gap-2">
              <Brain size={16} /> MLP 训练模式
            </h1>
            <p className="text-[9px] text-gray-500">逐指独立运动采集 · 伪标签训练 · 权重导出</p>
          </div>
        </div>
        {phase === 'collect' && (
          <div className="flex items-center gap-2">
            <span className="text-[9px] text-gray-400">
              {completedSteps}/{totalSteps} 已完成
            </span>
            <button
              onClick={resetCollection}
              className="text-[8px] text-gray-500 hover:text-gray-300 transition-colors"
            >
              重置
            </button>
          </div>
        )}
      </div>

      {/* ─── Phase: Collect ─── */}
      {phase === 'collect' && (
        <div className="flex-1 flex flex-col">
          {/* Step progress */}
          <div className="px-4 py-3 bg-gray-900/50 border-b border-gray-800/50">
            <div className="flex items-center gap-1">
              {TRAINING_STEPS.map((step, idx) => {
                const isComplete = step === 'rest' ? !!collectedSegments.rest : !!collectedSegments[step];
                const isCurrent = idx === currentStep;
                return (
                  <div key={step} className="flex items-center gap-1">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[8px] font-bold border transition-colors
                      ${isComplete ? 'bg-green-600 border-green-500 text-white' :
                        isCurrent ? 'bg-purple-600/30 border-purple-500 text-purple-300' :
                        'bg-gray-800 border-gray-700 text-gray-600'}`}>
                      {isComplete ? <CheckCircle size={10} /> : idx + 1}
                    </div>
                    {idx < TRAINING_STEPS.length - 1 && (
                      <div className={`w-4 h-px ${isComplete ? 'bg-green-600' : 'bg-gray-700'}`} />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Recording popup */}
          <div className="flex-1 flex items-center justify-center p-6">
            <div className="bg-gray-900 border border-gray-700 rounded-2xl p-8 max-w-lg w-full text-center shadow-2xl">
              {/* Icon */}
              <div className={`w-16 h-16 rounded-full mx-auto mb-6 flex items-center justify-center
                ${isRecording ? 'bg-red-500/20 border-2 border-red-500/50' :
                  isRest ? 'bg-gray-700/50 border-2 border-gray-600' :
                  'bg-purple-500/20 border-2 border-purple-500/50'}`}>
                {isRecording ? (
                  <div className="w-5 h-5 rounded-full bg-red-500 animate-pulse" />
                ) : isRest ? (
                  <Circle size={24} className="text-gray-400" />
                ) : (
                  <Brain size={24} className="text-purple-400" />
                )}
              </div>

              {/* Step title */}
              <h2 className="text-xl font-bold mb-2">
                {isRest ? '静止基线采集' : `${fingerInfo?.label || ''} 独立运动采集`}
              </h2>

              {/* Instruction */}
              <p className="text-gray-400 text-sm mb-6 leading-relaxed">
                {isRest
                  ? '保持手部完全静止，系统将记录传感器的零位基线。'
                  : fingerInfo?.instruction}
              </p>

              {/* Channel info */}
              {!isRest && fingerInfo && (
                <div className="flex flex-wrap gap-1.5 justify-center mb-6">
                  {fingerInfo.channels.map(ch => (
                    <span key={ch} className="px-2 py-0.5 bg-purple-500/10 border border-purple-500/20 rounded text-[9px] text-purple-300 font-mono">
                      CH{ch + 1}
                    </span>
                  ))}
                </div>
              )}

              {/* Progress bar */}
              {isRecording && (
                <div className="mb-6">
                  <div className="w-full bg-gray-800 h-2.5 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-red-500 transition-all duration-100 rounded-full"
                      style={{ width: `${Math.min(100, (frameCount / TARGET_FRAMES) * 100)}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-gray-500 mt-2 font-mono">
                    采样中: {frameCount} / {TARGET_FRAMES} 帧 ({(frameCount / SAMPLE_RATE).toFixed(1)}s / {(TARGET_FRAMES / SAMPLE_RATE).toFixed(0)}s)
                  </p>
                </div>
              )}

              {/* Buttons */}
              <div className="flex gap-3 justify-center mb-4">
                <button
                  onClick={onBack}
                  className="px-5 py-2.5 rounded-lg bg-gray-800 text-gray-300 text-sm border border-gray-700 hover:bg-gray-700 transition-colors"
                >
                  返回
                </button>
                {isRecording ? (
                  <button
                    onClick={stopRecording}
                    disabled={frameCount < 10}
                    className="px-6 py-2.5 rounded-lg bg-red-600 text-white font-bold text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-red-500 transition-colors"
                  >
                    停止采集
                  </button>
                ) : (
                  <button
                    onClick={startRecording}
                    disabled={!connected}
                    className="px-6 py-2.5 rounded-lg bg-purple-600 text-white font-bold text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-purple-500 transition-colors"
                  >
                    开始采集
                  </button>
                )}
              </div>

              {/* Export data button in collect phase */}
              {!isRecording && Object.keys(collectedSegments).length > 0 && (
                <div>
                  <button
                    onClick={exportData}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-gray-800/80 text-gray-400 text-[10px] hover:text-gray-200 hover:bg-gray-800 transition-colors border border-gray-700/50"
                  >
                    <Download size={12} /> 导出已采集数据
                  </button>
                </div>
              )}

              {!connected && !isRecording && (
                <p className="text-[9px] text-red-400 mt-4">请先连接传感器串口</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ─── Phase: Train ─── */}
      {phase === 'train' && (
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl p-8 max-w-lg w-full shadow-2xl">
            <div className="text-center mb-6">
              <Brain size={32} className="text-purple-400 mx-auto mb-4" />
              <h2 className="text-xl font-bold mb-2">采集完成，准备训练</h2>
              <p className="text-gray-400 text-sm">
                已采集 {Object.keys(collectedSegments).length} 个动作段，
                基线已{baseline.length > 0 ? '' : '未'}计算
              </p>
            </div>

            {/* Collected data summary */}
            <div className="bg-gray-800/50 rounded-xl p-4 mb-6 space-y-2">
              {TRAINING_STEPS.map(step => {
                const frames = collectedSegments[step]?.length || 0;
                const isRest = step === 'rest';
                return (
                  <div key={step} className="flex items-center justify-between text-[10px]">
                    <span className="text-gray-400">
                      {isRest ? '静止基线' : FINGER_CHANNELS[step]?.label || step}
                    </span>
                    <span className={`font-mono ${frames > 0 ? 'text-green-400' : 'text-gray-600'}`}>
                      {frames > 0 ? `${frames} 帧 (${(frames / SAMPLE_RATE).toFixed(1)}s)` : '未采集'}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Training config */}
            <div className="space-y-4 mb-6">
              <div>
                <div className="flex justify-between text-[10px] mb-1">
                  <span className="text-gray-400">训练轮数 (Epochs)</span>
                  <span className="font-mono text-purple-400">{epochs}</span>
                </div>
                <input
                  type="range"
                  min="100"
                  max="1000"
                  step="50"
                  value={epochs}
                  onChange={e => setEpochs(parseInt(e.target.value))}
                  className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
              </div>
              <div>
                <div className="flex justify-between text-[10px] mb-1">
                  <span className="text-gray-400">学习率 (LR)</span>
                  <span className="font-mono text-purple-400">{learningRate}</span>
                </div>
                <input
                  type="range"
                  min="0.0001"
                  max="0.01"
                  step="0.0001"
                  value={learningRate}
                  onChange={e => setLearningRate(parseFloat(e.target.value))}
                  className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
              </div>
            </div>

            {/* Training progress */}
            {training && trainingProgress && (
              <div className="bg-gray-800/50 rounded-xl p-4 mb-6">
                <div className="flex items-center gap-2 mb-3">
                  <Loader size={14} className="text-purple-400 animate-spin" />
                  <span className="text-[10px] text-purple-300 font-bold">训练中...</span>
                </div>
                <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden mb-2">
                  <div
                    className="h-full bg-purple-500 transition-all duration-300"
                    style={{ width: `${(trainingProgress.epoch / epochs) * 100}%` }}
                  />
                </div>
                <div className="grid grid-cols-3 gap-2 text-[9px] font-mono">
                  <div>
                    <div className="text-gray-500">Epoch</div>
                    <div className="text-gray-300">{trainingProgress.epoch}/{epochs}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Train Loss</div>
                    <div className="text-cyan-400">{trainingProgress.trainLoss.toFixed(6)}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Val Loss</div>
                    <div className="text-yellow-400">{trainingProgress.valLoss.toFixed(6)}</div>
                  </div>
                </div>
              </div>
            )}

            {trainingError && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 mb-6 flex items-center gap-2">
                <AlertTriangle size={14} className="text-red-400" />
                <span className="text-[10px] text-red-400">{trainingError}</span>
              </div>
            )}

            {/* Buttons */}
            <div className="flex gap-3">
              <button
                onClick={resetCollection}
                className="px-4 py-2.5 rounded-lg bg-gray-800 text-gray-300 text-sm border border-gray-700 hover:bg-gray-700 transition-colors"
              >
                重新采集
              </button>
              <button
                onClick={exportData}
                className="px-4 py-2.5 rounded-lg bg-gray-800 text-gray-300 text-sm border border-gray-700 hover:bg-gray-700 transition-colors flex items-center gap-1.5"
              >
                <Download size={12} /> 导出数据
              </button>
              <button
                onClick={handleTrain}
                disabled={training}
                className="flex-1 px-4 py-2.5 rounded-lg bg-purple-600 text-white font-bold text-sm hover:bg-purple-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {training ? '训练中...' : '开始训练'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ─── Phase: Done ─── */}
      {phase === 'done' && (
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="bg-gray-900 border border-gray-700 rounded-2xl p-8 max-w-lg w-full shadow-2xl text-center">
            <CheckCircle size={48} className="text-green-400 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-2">训练完成</h2>
            <p className="text-gray-400 text-sm mb-6">
              MLP 权重已生成，可选择应用或导出
            </p>

            {trainingProgress && (
              <div className="bg-gray-800/50 rounded-xl p-4 mb-6">
                <div className="grid grid-cols-2 gap-4 text-[10px]">
                  <div>
                    <div className="text-gray-500 mb-1">最终训练损失</div>
                    <div className="text-cyan-400 font-mono text-lg">{trainingProgress.trainLoss.toFixed(6)}</div>
                  </div>
                  <div>
                    <div className="text-gray-500 mb-1">最终验证损失</div>
                    <div className="text-yellow-400 font-mono text-lg">{trainingProgress.valLoss.toFixed(6)}</div>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-3">
              <button
                onClick={applyWeights}
                className="w-full px-6 py-3 rounded-lg bg-purple-600 text-white font-bold text-sm hover:bg-purple-500 transition-colors flex items-center justify-center gap-2"
              >
                <Brain size={16} /> 应用权重到解耦器
              </button>

              <div className="pt-2 border-t border-gray-800">
                <p className="text-[9px] text-gray-500 mb-3 text-center">导出训练结果</p>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={exportWeights}
                    className="flex flex-col items-center justify-center gap-1 px-4 py-3 rounded-xl bg-gray-800/80 text-gray-300 text-sm border border-gray-700/50 hover:bg-gray-800 hover:border-gray-600 transition-colors"
                  >
                    <Brain size={16} className="text-purple-400" />
                    <span className="text-[10px] font-bold">导出 MLP 权重</span>
                    <span className="text-[8px] text-gray-500">mlp-weights.json</span>
                  </button>
                  <button
                    onClick={exportData}
                    className="flex flex-col items-center justify-center gap-1 px-4 py-3 rounded-xl bg-gray-800/80 text-gray-300 text-sm border border-gray-700/50 hover:bg-gray-800 hover:border-gray-600 transition-colors"
                  >
                    <Download size={16} className="text-cyan-400" />
                    <span className="text-[10px] font-bold">导出采集数据</span>
                    <span className="text-[8px] text-gray-500">mlp-training-data.json</span>
                  </button>
                </div>
              </div>

              <button
                onClick={resetCollection}
                className="w-full px-4 py-2 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
              >
                重新采集训练
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
