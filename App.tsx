
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Environment, ContactShadows } from '@react-three/drei';
import {
  Cpu,
  Activity,
  RefreshCw,
  CheckCircle,
  XCircle,
  Sliders,
  ArrowLeftRight,
  Zap,
  Info,
  Brain,
  Waves,
  Download,
  Upload,
  ChevronDown,
  ChevronRight
} from 'lucide-react';
import { HandModel } from './components/HandModel';
import { serialService } from './services/serialService';
import { revo2Service } from './services/revo2Service';
import { creepFilter } from './services/creepFilter';
import { mlpDecoupler } from './services/mlpDecoupler';
import { creepCalibrator } from './services/creepCalibration';
import TrainingMode from './TrainingMode';
import {
  SensorReadings,
  HandCalibration,
  CalibrationStep,
  CalibrationRange,
  SENSOR_MAP,
  HandSide,
  CARDProcessingResult,
  CreepModelParams
} from './types';

const TOTAL_SENSORS = 12;
const INITIAL_RANGES: CalibrationRange[] = Array(TOTAL_SENSORS).fill(null).map(() => ({ 
  min: 0, 
  max: 4095, 
  exponent: 1.0 
}));

export default function App() {
  // UI State
  const [showPanel, setShowPanel] = useState(true);
  const [ports, setPorts] = useState<any[]>([]);
  const [connectionMode, setConnectionMode] = useState<'WIRED' | 'WIRELESS'>('WIRED');
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({});

  // Connection State
  const [connected, setConnected] = useState(false);
  const [selectedPortIndex, setSelectedPortIndex] = useState(0);

  // Revo2 Connection State
  const [revo2Connected, setRevo2Connected] = useState(false);
  const [revo2WsConnected, setRevo2WsConnected] = useState(false);
  const [revo2AddressInput, setRevo2AddressInput] = useState('0x0110');

  // Data
  const [rawData, setRawData] = useState<number[]>(Array(TOTAL_SENSORS).fill(0));
  const [filteredData, setFilteredData] = useState<number[]>(Array(TOTAL_SENSORS).fill(0));
  const [manualMode, setManualMode] = useState(false);
  const [manualData, setManualData] = useState<number[]>(Array(TOTAL_SENSORS).fill(2048));
  const [manualOpposition, setManualOpposition] = useState(0);
  const [manualDIP, setManualDIP] = useState({ index: 0, middle: 0, ring: 0, pinky: 0 });

  // Calibration State
  const [calibrations, setCalibrations] = useState<HandCalibration>({ 
    ranges: INITIAL_RANGES, 
    isCalibrated: false 
  });
  const [calStep, setCalStep] = useState<CalibrationStep>(CalibrationStep.IDLE);
  const [calBuffer, setCalBuffer] = useState<number[][]>([]);
  const [completeCalibrationMode, setCompleteCalibrationMode] = useState(false);

  // CARD Processing State
  const [cardResult, setCardResult] = useState<CARDProcessingResult>({
    raw: Array(TOTAL_SENSORS).fill(0),
    compensated: Array(TOTAL_SENSORS).fill(0),
    decoupled: Array(TOTAL_SENSORS).fill(0),
    filtered: Array(TOTAL_SENSORS).fill(0),
    normalized: Array(TOTAL_SENSORS).fill(0),
    derived: {
      thumbOpposition: 0,
      indexDIP: 0,
      middleDIP: 0,
      ringDIP: 0,
      pinkyDIP: 0,
      indexAbduction: 0,
      middleAbduction: 0,
      ringAbduction: 0,
      pinkyAbduction: 0,
      indexWeightedMCP: 0,
      middleWeightedMCP: 0,
    },
  });
  const [creepEnabled, setCreepEnabled] = useState(true);
  const [decouplerEnabled, setDecouplerEnabled] = useState(false);
  const [smoothingAlpha, setSmoothingAlpha] = useState(0.25);
  const dataHistoryRef = useRef<number[][]>([]);
  const [creepSubStep, setCreepSubStep] = useState<'HOLD_LOAD' | 'HOLD_RELEASE' | 'CYCLE'>('HOLD_LOAD');
  const creepLoadBufferRef = useRef<number[][]>([]);
  const creepUnloadBufferRef = useRef<number[][]>([]);
  const [creepCalibrated, setCreepCalibrated] = useState(false);
  const creepFileInputRef = useRef<HTMLInputElement>(null);
  const calFileInputRef = useRef<HTMLInputElement>(null);

  const [isRecording, setIsRecording] = useState(false);
  const recordingBufferRef = useRef<number[][]>([]);
  const serialDataCallbackRef = useRef<((data: number[]) => void) | null>(null);

  const [trainingMode, setTrainingMode] = useState(false);
  const [mlpWeightsLoaded, setMlpWeightsLoaded] = useState(false);
  const [mlpBaseline, setMlpBaseline] = useState<number[]>([]);
  const [mlpMaxDelta, setMlpMaxDelta] = useState<number[]>([]);
  const mlpWeightsFileRef = useRef<HTMLInputElement>(null);

  // --- Helpers ---
  const toggleSection = (key: string) => {
    setCollapsedSections(prev => ({ ...prev, [key]: !prev[key] }));
  };
  const isCollapsed = (key: string) => !!collapsedSections[key];

  const refreshPorts = useCallback(async () => {
    const availablePorts = await serialService.getAuthorizedPorts();
    setPorts(availablePorts);
  }, []);

  useEffect(() => {
    refreshPorts();
    if (navigator.serial) {
      navigator.serial.addEventListener('connect', refreshPorts);
      navigator.serial.addEventListener('disconnect', refreshPorts);
    }
    return () => {
      if (navigator.serial) {
        navigator.serial.removeEventListener('connect', refreshPorts);
        navigator.serial.removeEventListener('disconnect', refreshPorts);
      }
    }
  }, [refreshPorts]);

  const handleAddPort = async () => {
    const port = await serialService.requestAccess();
    if (port) {
      await refreshPorts();
      const updatedPorts = await serialService.getAuthorizedPorts();
      const idx = updatedPorts.findIndex(p => p === port);
      if (idx !== -1) setSelectedPortIndex(idx);
    }
  };

  // --- Logic: CARD Processing Pipeline ---
  const processWithCARD = useCallback((raw: number[], manual: boolean = false): CARDProcessingResult => {
    const sourceData = manual ? manualData : raw;
    
    mlpDecoupler.setEnabled(decouplerEnabled);

    const compensated = creepEnabled 
      ? creepFilter.process(sourceData)
      : sourceData;

    // MLP 在差值空间解耦：减去训练时的基线 → 归一化 → 解耦 → 反归一化 → 加回基线
    const baselines = mlpBaseline.length === 12 ? mlpBaseline : calibrations.ranges.map(r => r.min);
    const compensatedDeltas = compensated.map((v, i) => v - baselines[i]);
    
    // 归一化到 [-1, 1]（使用训练时的 maxDelta）
    const normalizedDeltas = mlpMaxDelta.length === 12 
      ? compensatedDeltas.map((v, i) => Math.max(-1, Math.min(1, v / mlpMaxDelta[i])))
      : compensatedDeltas;
    
    const decoupledNormalized = mlpDecoupler.process(normalizedDeltas);
    
    // 反归一化并加回基线
    const decoupled = mlpMaxDelta.length === 12
      ? decoupledNormalized.map((v, i) => v * mlpMaxDelta[i] + baselines[i])
      : decoupledNormalized.map((v, i) => v + baselines[i]);

    const filtered = cardResult.filtered.map((prev, i) => {
      return smoothingAlpha * decoupled[i] + (1 - smoothingAlpha) * prev;
    });

    const normalized = filtered.map((val, idx) => {
      const range = calibrations.ranges[idx];
      let diff = range.max - range.min;
      
      if (Math.abs(diff) < 10) return 0; 
      
      let n = (val - range.min) / diff;
      n = Math.max(0, Math.min(1, n));
      
      const deadzone = 0.05; 
      if (n < deadzone) n = 0;
      else n = (n - deadzone) / (1 - deadzone);
      
      return Math.pow(n, range.exponent);
    });

    const calcDIP = (pipIdx: number) => normalized[pipIdx] * 1.0;

    return {
      raw: sourceData,
      compensated,
      decoupled,
      filtered,
      normalized,
      derived: {
        thumbOpposition: manual ? manualOpposition : Math.max(0, Math.min(1, 1 - normalized[SENSOR_MAP.THUMB_SPREAD])),
        indexDIP: manual ? manualDIP.index : calcDIP(SENSOR_MAP.INDEX_PIP),
        middleDIP: manual ? manualDIP.middle : calcDIP(SENSOR_MAP.MIDDLE_PIP),
        ringDIP: manual ? manualDIP.ring : calcDIP(SENSOR_MAP.RING_PIP),
        pinkyDIP: manual ? manualDIP.pinky : calcDIP(SENSOR_MAP.PINKY_PIP),
        indexAbduction: normalized[SENSOR_MAP.INDEX_MIDDLE_SPREAD] * 1.2,
        middleAbduction: normalized[SENSOR_MAP.MIDDLE_RING_SPREAD] * 0.6,
        ringAbduction: normalized[SENSOR_MAP.MIDDLE_RING_SPREAD] * (-0.6),
        pinkyAbduction: normalized[SENSOR_MAP.RING_PINKY_SPREAD] * (-1.3),
        thumbWeightedMCP: (normalized[SENSOR_MAP.THUMB_IP] * 0.4 + normalized[SENSOR_MAP.THUMB_MCP] * 0.6),
        indexWeightedMCP: (normalized[SENSOR_MAP.INDEX_MCP] * 0.7 + normalized[SENSOR_MAP.INDEX_PIP] * 0.3),
        middleWeightedMCP: (normalized[SENSOR_MAP.MIDDLE_MCP] * 0.7 + normalized[SENSOR_MAP.MIDDLE_PIP] * 0.3),
      },
    };
  }, [manualData, calibrations, creepEnabled, decouplerEnabled, smoothingAlpha, cardResult.filtered, manualOpposition, manualDIP, mlpBaseline, mlpMaxDelta]);

  const processedData = useMemo((): SensorReadings => {
    return {
      raw: cardResult.raw,
      normalized: cardResult.normalized,
      derived: cardResult.derived,
    };
  }, [cardResult]);

  // --- Serial Handler ---
  const handleSerialData = useCallback((data: number[]) => {
    const now = Date.now();
    if (!(window as any).lastSerialUpdate) (window as any).lastSerialUpdate = 0;
    
    if (calStep !== CalibrationStep.IDLE) {
      setCalBuffer(prev => [...prev, data]);
    }

    if (now - (window as any).lastSerialUpdate < 25) {
       return;
    }
    (window as any).lastSerialUpdate = now;

    setRawData(data);

    dataHistoryRef.current.push(data);
    if (dataHistoryRef.current.length > 500) {
      dataHistoryRef.current.shift();
    }

    if (isRecording) {
      recordingBufferRef.current.push([...data]);
    }

    const result = processWithCARD(data, false);
    setCardResult(result);
  }, [calStep, processWithCARD, isRecording]);

  useEffect(() => {
    serialDataCallbackRef.current = handleSerialData;
  }, [handleSerialData]);

  const stableSerialDataCallback = useCallback((data: number[]) => {
    serialDataCallbackRef.current?.(data);
  }, []);

  // --- Manual Mode CARD Processing ---
  useEffect(() => {
    if (manualMode) {
      const result = processWithCARD(manualData, true);
      setCardResult(result);
    }
  }, [manualData, manualMode, manualOpposition, manualDIP, processWithCARD]);

  // --- Reset CARD State on Mode Change ---
  useEffect(() => {
    if (!manualMode) {
      creepFilter.reset();
      mlpDecoupler.reset();
    }
  }, [manualMode]);

  // --- Load persisted creep params on mount ---
  useEffect(() => {
    const saved = localStorage.getItem('creepParams');
    if (saved) {
      try {
        const params = JSON.parse(saved) as CreepModelParams[];
        // 兼容旧格式 {tau, alpha, hysteresisGap} → 新格式 {tauLoad, tauUnload, alphaLoad, alphaUnload, hysteresisGap}
        const migrated = params.map(p => {
          if ('tauLoad' in p) return p;
          const old = p as any;
          return {
            tauLoad: old.tau ?? 0.2,
            tauUnload: (old.tau ?? 0.2) * 1.5,
            alphaLoad: old.alpha ?? 0.08,
            alphaUnload: (old.alpha ?? 0.08) * 0.8,
            hysteresisGap: old.hysteresisGap ?? 0.02,
          } as CreepModelParams;
        });
        creepFilter.setAllParams(migrated);
        setCreepCalibrated(true);
      } catch {
        // 解析失败则使用默认参数
      }
    }
    // Load persisted MLP weights
    const savedWeights = localStorage.getItem('mlpWeights');
    if (savedWeights) {
      try {
        const data = JSON.parse(savedWeights);
        const weights = data.weights || data;
        if (weights.input && weights.hidden && weights.output) {
          mlpDecoupler.setWeights(weights);
          setMlpWeightsLoaded(true);
          if (Array.isArray(data.baseline) && data.baseline.length === 12) {
            setMlpBaseline(data.baseline);
          }
          if (Array.isArray(data.maxDelta) && data.maxDelta.length === 12) {
            setMlpMaxDelta(data.maxDelta);
          }
        }
      } catch {
        // ignore
      }
    }
  }, []);

  // --- Revo2 Updater ---
  // Throttle the Revo2 update to max 30 FPS to avoid flooding the bridge
  useEffect(() => {
    if (revo2Connected || revo2WsConnected) {
      
      const now = Date.now();
      // Ensure local ref exists for throttling
      if (!(window as any).lastRevo2Update) (window as any).lastRevo2Update = 0;
      if (now - (window as any).lastRevo2Update < 30) return; // ~33Hz max
      (window as any).lastRevo2Update = now;

      // Revo2 takes 0~1.0 normalized value, processedData has accurate weighted abstractions.
      const ipVal = processedData.normalized[SENSOR_MAP.THUMB_IP];
      const cmcTuck = processedData.normalized[SENSOR_MAP.THUMB_MCP];
      const mcpVal = (ipVal * 0.4) + (cmcTuck * 0.6);

      const thumbFlexion = mcpVal;
      const thumbRotation = cmcTuck;
      const index = processedData.derived.indexWeightedMCP;
      const middle = processedData.derived.middleWeightedMCP;
      const ring = processedData.normalized[SENSOR_MAP.RING_PIP];
      const pinky = processedData.normalized[SENSOR_MAP.PINKY_PIP];

      if (revo2WsConnected) {
        revo2Service.sendFingerPositionsWs(thumbFlexion, thumbRotation, index, middle, ring, pinky);
      } else if (revo2Connected) {
        const address = parseInt(revo2AddressInput, revo2AddressInput.startsWith('0x') || revo2AddressInput.startsWith('0X') ? 16 : 10);
        if (isNaN(address)) return;

        revo2Service.sendFingerPositions(
          thumbFlexion,
          thumbRotation,
          index,
          middle,
          ring,
          pinky,
          address
        ).catch(err => console.error("Revo2 update error", err));
      }
    }
  }, [processedData, revo2Connected, revo2WsConnected, revo2AddressInput]);

  const handleConnectionToggle = async () => {
    if (connected) {
      await serialService.disconnect('RIGHT');
      setConnected(false);
      return;
    }

    const port = ports[selectedPortIndex];
    if (port) {
      const success = await serialService.connectToPort('RIGHT', port, stableSerialDataCallback);
      setConnected(success);
    } else {
      await handleAddPort();
    }
  };

  const handleRevo2Toggle = async () => {
    if (revo2Connected) {
      await revo2Service.disconnect();
      setRevo2Connected(false);
    } else {
      const success = await revo2Service.requestAndConnect();
      setRevo2Connected(success);
    }
  };

  const handleRevo2WsToggle = async () => {
    if (revo2WsConnected) {
      revo2Service.disconnectWs();
      setRevo2WsConnected(false);
    } else {
      const success = await revo2Service.connectWebSocket();
      if (!success) {
        alert("无法连接到本地 Python 桥接服务。请确保已在本地运行脚本。");
      }
      setRevo2WsConnected(success);
    }
  };

  // --- Calibration Logic ---
  const MIN_CAL_FRAMES = 30; // 每步标定至少需要的帧数

  const startCalibration = (completeMode: boolean = false) => {
    setCalStep(CalibrationStep.RELAX);
    setCalBuffer([]);
    setCompleteCalibrationMode(completeMode);
  };

  // --- Data Recording Logic ---
  const startRecording = () => {
    recordingBufferRef.current = [];
    setIsRecording(true);
  };

  const stopRecording = () => {
    setIsRecording(false);
    const data = recordingBufferRef.current;
    if (data.length === 0) return;
    const csv = data.map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sensor-recording-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const startCreepCalibration = () => {
    setCalStep(CalibrationStep.CREEP);
    setCreepSubStep('HOLD_LOAD');
    setCalBuffer([]);
  };

  const exportCreepParams = () => {
    const params = creepFilter.getAllParams();
    const data = JSON.stringify(params, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `creep-params-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const importCreepParams = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const params = JSON.parse(event.target?.result as string) as CreepModelParams[];
        if (Array.isArray(params) && params.length === 12) {
          creepFilter.setAllParams(params);
          localStorage.setItem('creepParams', JSON.stringify(params));
          setCreepCalibrated(true);
        } else {
          alert('参数格式无效：需要 12 通道的蠕变参数数组。');
        }
      } catch {
        alert('文件解析失败，请确认是有效的 JSON 文件。');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const exportCalParams = () => {
    const data = JSON.stringify(calibrations, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `calibration-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const importCalParams = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const cal = JSON.parse(event.target?.result as string);
        if (cal && Array.isArray(cal.ranges) && cal.ranges.length === 12 && typeof cal.isCalibrated === 'boolean') {
          setCalibrations(cal);
        } else {
          alert('参数格式无效：需要包含 12 通道 ranges 和 isCalibrated 字段的标定对象。');
        }
      } catch {
        alert('文件解析失败，请确认是有效的 JSON 文件。');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const importMLPWeights = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target?.result as string);
        const weights = data.weights || data;
        if (weights.input && weights.hidden && weights.output &&
            Array.isArray(weights.input.weights) && Array.isArray(weights.hidden.weights) && Array.isArray(weights.output.weights)) {
          mlpDecoupler.setWeights(weights);
          mlpDecoupler.setEnabled(true);
          setDecouplerEnabled(true);
          setMlpWeightsLoaded(true);
          if (Array.isArray(data.baseline) && data.baseline.length === 12) {
            setMlpBaseline(data.baseline);
          }
          if (Array.isArray(data.maxDelta) && data.maxDelta.length === 12) {
            setMlpMaxDelta(data.maxDelta);
          }
          localStorage.setItem('mlpWeights', JSON.stringify(data));
        } else {
          alert('参数格式无效：需要包含 input/hidden/output 三层权重的 MLP 权重文件。');
        }
      } catch {
        alert('文件解析失败，请确认是有效的 JSON 文件。');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const nextCalibrationStep = () => {
    const buffer = calBuffer;
    // A. 稳定帧过滤：剔除前20%帧，只取稳定部分
    const stableStart = Math.floor(buffer.length * 0.2);
    const stableBuffer = buffer.slice(stableStart);
    const snapshot = stableBuffer.length > 0 
      ? stableBuffer[0].map((_, colIndex) => stableBuffer.reduce((acc, row) => acc + row[colIndex], 0) / stableBuffer.length)
      : rawData;

    let newRanges = calibrations.ranges.map(r => ({ ...r }));
    const spreadIndices = [2, 4, 7, 10];
    
    if (calStep === CalibrationStep.RELAX) {
       // F. 自然放松态：取所有传感器的 min（最松弛状态）
       snapshot.forEach((val, idx) => {
          if (idx < newRanges.length) newRanges[idx].min = val;
       });
       setCalibrations({ ...calibrations, ranges: newRanges });
       setCalStep(CalibrationStep.FIST);
    } else if (calStep === CalibrationStep.FIST) {
       const flexionIndices = [0, 1, 3, 5, 6, 8, 9, 11];
       flexionIndices.forEach(idx => {
          if (idx < snapshot.length) newRanges[idx].max = snapshot[idx];
       });
       // D. 张开传感器max修正：握拳时指间皮肤牵引也会拉伸张开传感器
       spreadIndices.forEach(idx => {
          if (idx < snapshot.length && snapshot[idx] > newRanges[idx].max) {
             newRanges[idx].max = snapshot[idx];
          }
       });
       setCalibrations({ ...calibrations, ranges: newRanges });
       setCalStep(CalibrationStep.FLAT);
    } else if (calStep === CalibrationStep.FLAT) {
       // FLAT 仍用于修正 min（并拢状态可能比自然放松更紧，取较小值）
       snapshot.forEach((val, idx) => {
          if (idx < newRanges.length && val < newRanges[idx].min) {
             newRanges[idx].min = val;
          }
       });
       // FLAT 时张开传感器处于最小值（并拢），更新 min
       spreadIndices.forEach(idx => {
          if (idx < newRanges.length) newRanges[idx].min = Math.min(newRanges[idx].min, snapshot[idx] || newRanges[idx].min);
       });
       setCalibrations({ ...calibrations, ranges: newRanges });
       setCalStep(CalibrationStep.SPREAD);
    } else if (calStep === CalibrationStep.SPREAD) {
       spreadIndices.forEach(idx => {
          if (idx < snapshot.length && snapshot[idx] > newRanges[idx].max) {
             newRanges[idx].max = snapshot[idx];
          }
       });
       setCalibrations({ ...calibrations, ranges: newRanges, isCalibrated: true });
       // E. 重新标定后同步 MLP baseline
       if (mlpBaseline.length === 12) {
          setMlpBaseline(newRanges.map(r => r.min));
       }
       if (completeCalibrationMode) {
          setCalStep(CalibrationStep.CREEP);
          setCreepSubStep('HOLD_LOAD');
       } else {
          setCalStep(CalibrationStep.IDLE);
       }
    } else if (calStep === CalibrationStep.CREEP) {
       // 三步蠕变标定：HOLD_LOAD → HOLD_RELEASE → CYCLE
       if (creepSubStep === 'HOLD_LOAD') {
          creepLoadBufferRef.current = [...buffer];
          setCreepSubStep('HOLD_RELEASE');
          setCalBuffer([]);
          return;
       }
       if (creepSubStep === 'HOLD_RELEASE') {
          creepUnloadBufferRef.current = [...buffer];
          setCreepSubStep('CYCLE');
          setCalBuffer([]);
          return;
       }
       // CYCLE 阶段：使用加载+卸载双缓冲区拟合，循环数据估算滞回
       const loadBuffer = creepLoadBufferRef.current;
       const unloadBuffer = creepUnloadBufferRef.current;
       if (loadBuffer.length >= 30 && unloadBuffer.length >= 30) {
          const creepParams = creepCalibrator.calibrateAllChannelsFromBuffer(loadBuffer, unloadBuffer, 40);
          const hysteresisGap = buffer.length >= 10
             ? creepCalibrator.estimateHysteresisGap(buffer)
             : 0.02;
          creepParams.forEach((params, ch) => {
             if (params.rSquaredLoad < 0.7) {
                const transient = creepCalibrator.estimateFromTransient(
                   loadBuffer.map(d => d[ch]), true
                );
                creepFilter.setParams(ch, { ...transient, hysteresisGap });
             } else if (params.rSquaredUnload < 0.7) {
                const transient = creepCalibrator.estimateFromTransient(
                   unloadBuffer.map(d => d[ch]), false
                );
                creepFilter.setParams(ch, { ...params, ...transient, hysteresisGap });
             } else {
                creepFilter.setParams(ch, { ...params, hysteresisGap });
             }
          });
          const allParams = creepFilter.getAllParams();
          localStorage.setItem('creepParams', JSON.stringify(allParams));
          setCreepCalibrated(true);
       }
       if (completeCalibrationMode) {
          setCalStep(CalibrationStep.IDLE);
          setTrainingMode(true);
          setCompleteCalibrationMode(false);
       } else {
          setCalStep(CalibrationStep.IDLE);
       }
    }
    setCalBuffer([]);
  };

  const updateManual = (idx: number, val: number) => {
    setManualData(prev => {
        const n = [...prev];
        n[idx] = val;
        return n;
    });
  };

  const updateCurve = (idx: number, exponent: number) => {
    setCalibrations(prev => {
      const newRanges = prev.ranges.map((r, i) => i === idx ? { ...r, exponent } : r);
      return { ...prev, ranges: newRanges };
    });
  };

  const updateRangeMin = (idx: number, val: number) => {
    setCalibrations(prev => {
      const newRanges = prev.ranges.map((r, i) => i === idx ? { ...r, min: val } : r);
      return { ...prev, ranges: newRanges };
    });
  };

  const updateRangeMax = (idx: number, val: number) => {
    setCalibrations(prev => {
      const newRanges = prev.ranges.map((r, i) => i === idx ? { ...r, max: val } : r);
      return { ...prev, ranges: newRanges };
    });
  };

  const renderWeightedDisplay = (label: string, value: number) => {
    return (
      <div className="mb-3 bg-cyan-900/10 p-2 rounded-lg border border-cyan-800/20">
        <div className="flex justify-between text-[11px] mb-1.5 items-center">
          <span className="text-cyan-300 font-bold">{label}</span>
          <span className="font-mono text-[9px] text-cyan-400">
             <span className="w-8 text-right">CALC</span>
             <span className="mx-1 text-gray-700">|</span>
             <span className="w-8 text-right font-black">
               {(value * 100).toFixed(0)}%
             </span>
          </span>
        </div>
        <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden border border-cyan-900/30">
          <div className="h-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.5)] transition-all duration-300" style={{ width: `${value * 100}%` }} />
        </div>
        <div className="mt-1 text-[8px] text-gray-500 uppercase tracking-wider italic">基于指间关节数据加权计算</div>
      </div>
    );
  };

  const renderSensorControl = (label: string, idx: number) => {
    const range = calibrations.ranges[idx];
    const rawVal = manualMode ? manualData[idx] : rawData[idx];
    const normVal = cardResult.normalized[idx];
    const compensatedVal = cardResult.compensated[idx];

    return (
      <div key={idx} className="mb-3 group bg-gray-900/40 p-2 rounded-lg border border-gray-800/40">
        <div className="flex justify-between text-[11px] mb-1.5 items-center">
          <span className="text-gray-300 font-medium group-hover:text-cyan-200 transition-colors">{label}</span>
          <span className="font-mono text-[9px] text-gray-500 flex gap-2">
             <span className="text-gray-600 w-8 text-right">{rawVal.toFixed(0)}</span>
             <span className="text-gray-700">|</span>
             <span className={`w-8 text-right font-bold ${manualMode ? "text-purple-400" : "text-cyan-400"}`}>
               {(normVal * 100).toFixed(0)}%
             </span>
          </span>
        </div>
        
        {manualMode ? (
           <input type="range" min={range.min} max={range.max} value={manualData[idx]}
             onChange={(e) => updateManual(idx, parseInt(e.target.value))}
             className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500 mb-2" />
        ) : (
          <div className="w-full bg-gray-800 h-1 rounded-full overflow-hidden border border-gray-700/50 mb-2">
            <div className="h-full bg-cyan-500 transition-all duration-200" style={{ width: `${normVal * 100}%` }} />
          </div>
        )}

        <div className="flex items-center gap-2 mt-1 px-1">
           <Zap size={10} className={range.exponent !== 1 ? "text-yellow-500" : "text-gray-600"} />
           <input type="range" min="0.2" max="3.0" step="0.05" value={range.exponent}
             onChange={(e) => updateCurve(idx, parseFloat(e.target.value))}
             className="flex-1 h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-gray-500" />
           <span className="text-[9px] font-mono text-gray-500 w-8 text-right">k={range.exponent.toFixed(1)}</span>
        </div>
        
        <div className="flex items-center justify-between mt-2 pt-1 border-t border-gray-800/50 px-1">
           <div className="flex items-center gap-1.5 opacity-70 hover:opacity-100 transition-opacity">
             <span className="text-[8px] text-gray-500 font-mono">MIN</span>
             <input type="number" value={Math.round(range.min)} onChange={(e) => updateRangeMin(idx, parseInt(e.target.value) || 0)} className="w-[45px] bg-gray-900 text-gray-300 text-[9px] font-mono rounded px-1 py-0.5 outline-none text-center border border-gray-700/50 hover:border-gray-500 focus:border-cyan-500/50 transition-colors" title="手动微调最小极值" />
           </div>
           <div className="flex items-center gap-1.5 opacity-70 hover:opacity-100 transition-opacity">
             <span className="text-[8px] text-gray-500 font-mono">MAX</span>
             <input type="number" value={Math.round(range.max)} onChange={(e) => updateRangeMax(idx, parseInt(e.target.value) || 0)} className="w-[45px] bg-gray-900 text-gray-300 text-[9px] font-mono rounded px-1 py-0.5 outline-none text-center border border-gray-700/50 hover:border-gray-500 focus:border-cyan-500/50 transition-colors" title="手动微调最大极值" />
           </div>
        </div>
      </div>
    );
  };

  const renderDIPControl = (label: string, key: 'index' | 'middle' | 'ring' | 'pinky') => {
    const val = cardResult.derived[`${key}DIP` as keyof typeof cardResult.derived];

    return (
      <div className="mb-2 group bg-gray-900/20 p-2 rounded-lg border border-dashed border-gray-800/40">
        <div className="flex justify-between text-[11px] mb-1 items-center">
          <span className="text-purple-400/80 font-medium group-hover:text-purple-300 transition-colors">{label}</span>
          <span className="font-mono text-[10px] font-bold text-purple-400">{(val * 100).toFixed(0)}%</span>
        </div>
        {manualMode ? (
           <input type="range" min="0" max="1" step="0.01" value={manualDIP[key]}
             onChange={(e) => {
                setManualDIP(prev => ({ ...prev, [key]: parseFloat(e.target.value) }));
             }}
             className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-400" />
        ) : (
          <div className="w-full bg-gray-800/50 h-1 rounded-full overflow-hidden">
            <div className="h-full bg-purple-500/60" style={{ width: `${val * 100}%` }} />
          </div>
        )}
      </div>
    );
  };

  const getPortLabel = (port: any, index: number) => {
    const info = port.getInfo();
    const vid = info.usbVendorId?.toString(16).padStart(4, '0').toUpperCase();
    const pid = info.usbProductId?.toString(16).padStart(4, '0').toUpperCase();
    return vid ? `Dev ${index + 1} (${vid}:${pid})` : `Serial ${index + 1}`;
  };

  return (
    <div className="h-screen w-screen bg-gray-950 flex flex-col overflow-hidden text-gray-100 font-sans">
      <header className="h-14 border-b border-gray-800 flex items-center justify-between px-4 bg-gray-900/90 backdrop-blur z-50">
        <div className="flex items-center gap-3">
          <Cpu className="w-5 h-5 text-cyan-450" />
          <div className="flex items-baseline gap-2">
            <h1 className="font-bold text-lg tracking-tight text-gray-100">Lyokinn <span className="text-cyan-450">流金智感</span></h1>
          <span className="text-[10px] text-gray-500 font-mono border border-gray-700 px-1.5 rounded">V0.3 CARD</span>
          </div>
          <div className={`ml-4 text-[10px] font-bold px-2 py-0.5 rounded uppercase ${connected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
            {connected ? '设备已连接' : '未连接设备'}
          </div>
        </div>

        <div className="flex items-center gap-2">
            <div className="flex bg-gray-800/80 rounded-md p-0.5 border border-gray-700 mr-2 shadow-inner">
                <button 
                  onClick={() => setConnectionMode('WIRED')} 
                  disabled={connected}
                  className={`px-3 py-0.5 text-[9px] font-bold rounded transition-colors ${connectionMode === 'WIRED' ? 'bg-cyan-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}
                >
                  有线单手
                </button>
                <button 
                  onClick={() => setConnectionMode('WIRELESS')} 
                  disabled={connected}
                  className={`px-3 py-0.5 text-[9px] font-bold rounded transition-colors ${connectionMode === 'WIRELESS' ? 'bg-cyan-600 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}
                >
                  无线单手
                </button>
            </div>

            <div className="flex items-center bg-gray-800/50 rounded-md border border-cyan-700/30 p-0.5 group">
                <Zap size={10} className="ml-2 text-cyan-400" />
                <select value={selectedPortIndex} onChange={(e) => setSelectedPortIndex(Number(e.target.value))} disabled={connected}
                    className="bg-transparent text-[10px] text-gray-300 py-1 pl-1 outline-none min-w-[120px]">
                    {ports.length === 0 && <option value={0}>未发现设备</option>}
                    {ports.map((p, idx) => (<option key={idx} value={idx}>{getPortLabel(p, idx)}</option>))}
                </select>

                <button onClick={handleConnectionToggle} className={`ml-2 px-4 py-1 rounded text-[10px] font-bold border transition-all ${connected ? 'bg-red-900/30 text-red-400 border-red-800' : 'bg-cyan-600 text-white border-cyan-700 shadow-lg shadow-cyan-500/20'}`}>
                    {connected ? '断开输入' : '连接输入'}
                </button>
            </div>
            
            <div className="flex items-center bg-gray-800/50 rounded-md border border-purple-700/30 p-0.5 group">
                <button 
                  onClick={handleRevo2WsToggle} 
                  className={`px-3 py-1 mr-1 rounded text-[10px] font-bold border transition-all ${revo2WsConnected ? 'bg-green-900/40 text-green-400 border-green-800' : 'bg-gray-800 text-green-400 border-gray-700 hover:bg-green-900/20'}`}
                  title="需在本地运行 Python SDK 桥接脚本"
                >
                    {revo2WsConnected ? '关闭 SDK 网桥' : '通过官方 SDK 桥接控制'}
                </button>
                <div className="w-px h-4 bg-gray-700 mx-1"></div>
                <div className="flex items-center px-2 py-0.5 mx-1 rounded border border-gray-700 bg-gray-900 overflow-hidden">
                    <span className="text-[9px] text-gray-500 mr-1 select-none" title="未通过SDK时的直接寄存器地址">Reg:</span>
                    <input 
                      type="text" 
                      value={revo2AddressInput} 
                      onChange={e => setRevo2AddressInput(e.target.value)} 
                      disabled={revo2Connected}
                      className="bg-transparent text-[10px] text-purple-400 font-mono w-10 outline-none disabled:opacity-50" 
                      placeholder="0x0110" 
                    />
                </div>
                <button 
                  onClick={handleRevo2Toggle} 
                  className={`px-3 py-1 rounded text-[10px] font-bold border transition-all ${revo2Connected ? 'bg-purple-900/30 text-purple-400 border-purple-800' : 'bg-gray-800 text-purple-400 border-gray-700 hover:bg-purple-900/20'}`}
                >
                    {revo2Connected ? '断开 Modbus' : '直接写 Modbus'}
                </button>
            </div>

            <button onClick={() => setShowPanel(!showPanel)} className={`p-1.5 ml-2 rounded ${showPanel ? 'bg-cyan-900/30 text-cyan-400' : 'bg-gray-800 text-gray-400'}`}><Activity size={18} /></button>
        </div>
      </header>

      <div className="flex-1 flex relative overflow-hidden">
        <div className="flex-1 min-w-0 relative bg-gray-900">
           <Canvas shadows camera={{ position: [0, 2, 10], fov: 40 }}>
             <fog attach="fog" args={['#0d1117', 10, 60]} />
             <ambientLight intensity={1.5} />
             <directionalLight position={[10, 10, 10]} intensity={2.5} castShadow />
             <Environment preset="city" />
             <HandModel 
               data={processedData} 
               calibration={calibrations}
               side="RIGHT" 
               position={[0, -2, 0]} 
             />
             <ContactShadows position={[0, -2, 0]} opacity={0.6} scale={15} blur={1.5} far={4} color="#000000" />
             <Grid args={[40, 40]} cellSize={0.5} cellThickness={0.5} sectionSize={3} fadeDistance={50} sectionColor="#00e5ff" cellColor="#1a202c" position={[0, -2, 0]} />
             <OrbitControls makeDefault minPolarAngle={0} maxPolarAngle={Math.PI / 1.8} />
           </Canvas>

           {calStep !== CalibrationStep.IDLE && (
             <div className="absolute inset-0 bg-black/80 backdrop-blur-md flex items-center justify-center z-50 p-4">
                <div className="bg-gray-900 border border-gray-700 p-6 rounded-2xl max-w-lg w-full shadow-2xl">
                   {/* Mode badge */}
                   <div className="flex items-center justify-center mb-4">
                     {completeCalibrationMode ? (
                       <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-gradient-to-r from-cyan-600/30 via-yellow-600/30 to-purple-600/30 border border-cyan-600/30">
                         <Zap size={12} className="text-cyan-400" />
                         <span className="text-[9px] font-bold text-white">完整标定模式</span>
                       </div>
                     ) : (
                       <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-600/20 border border-cyan-600/30">
                         <Sliders size={12} className="text-cyan-400" />
                         <span className="text-[9px] font-bold text-cyan-300">快速标定模式</span>
                       </div>
                     )}
                   </div>

                   <div className="flex items-center justify-center gap-3 mb-5">
                     {calStep === CalibrationStep.CREEP ? (
                       <Waves className="w-10 h-10 text-yellow-400 animate-pulse" />
                     ) : (
                       <RefreshCw className="w-10 h-10 text-cyan-450 animate-spin-slow" />
                     )}
                     <div className="text-left">
                       <h2 className="text-xl font-bold text-white">
                         {calStep === CalibrationStep.CREEP ? '蠕变特性标定' : '右手传感器标定'}
                       </h2>
                       <p className="text-[10px] text-gray-500 uppercase tracking-wider">
                         {completeCalibrationMode
                           ? (calStep === CalibrationStep.CREEP ? '步骤 5 / 6' :
                              `步骤 ${calStep === CalibrationStep.RELAX ? '1' : calStep === CalibrationStep.FIST ? '2' : calStep === CalibrationStep.FLAT ? '3' : '4'} / 6`)
                           : `步骤 ${calStep === CalibrationStep.RELAX ? '1' : calStep === CalibrationStep.FIST ? '2' : calStep === CalibrationStep.FLAT ? '3' : '4'} / 4`
                         }
                       </p>
                     </div>
                   </div>

                   {/* Complete mode: 6-step flow indicator */}
                   {completeCalibrationMode && (
                     <div className="bg-gray-800/50 rounded-xl p-4 mb-5">
                        <div className="flex justify-between items-center mb-2">
                           <span className="text-[9px] text-gray-400 font-bold uppercase">完整标定流程</span>
                           <span className="text-[8px] text-cyan-400 font-mono">自动流转</span>
                        </div>
                        <div className="flex items-center justify-between">
                          {[
                            { step: CalibrationStep.RELAX, label: '放松' },
                            { step: CalibrationStep.FIST, label: '握拳' },
                            { step: CalibrationStep.FLAT, label: '伸直' },
                            { step: CalibrationStep.SPREAD, label: '张开' },
                            { step: CalibrationStep.CREEP, label: '蠕变' },
                            { step: 'TRAIN', label: '训练' },
                          ].map((item, idx, arr) => {
                            const stepOrder = [CalibrationStep.RELAX, CalibrationStep.FIST, CalibrationStep.FLAT, CalibrationStep.SPREAD, CalibrationStep.CREEP];
                            const currentIdx = stepOrder.indexOf(calStep as any);
                            const thisIdx = stepOrder.indexOf(item.step as any);
                            const isPast = thisIdx >= 0 && currentIdx > thisIdx;
                            const isCurrent = calStep === item.step;
                            const isTrain = item.step === 'TRAIN';
                            const isCreep = item.step === CalibrationStep.CREEP;
                            return (
                              <React.Fragment key={idx}>
                                <div className="flex flex-col items-center gap-1">
                                  <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[7px] font-bold ${
                                    isTrain ? 'bg-purple-600/30 text-purple-400 border border-purple-600/50' :
                                    isCreep && isCurrent ? 'bg-yellow-500 text-white ring-2 ring-yellow-400/50' :
                                    isCreep && isPast ? 'bg-yellow-600 text-white' :
                                    isCurrent ? 'bg-cyan-600 text-white ring-2 ring-cyan-400/50' :
                                    isPast ? 'bg-green-600 text-white' :
                                    'bg-gray-700 text-gray-400'
                                  }`}>{idx + 1}</div>
                                  <span className="text-[6px] text-gray-500">{item.label}</span>
                                </div>
                                {idx < arr.length - 1 && (
                                  <div className={`w-5 h-0.5 ${isPast ? (item.step === CalibrationStep.CREEP ? 'bg-yellow-600' : 'bg-green-600') : 'bg-gray-700'}`}></div>
                                )}
                              </React.Fragment>
                            );
                          })}
                        </div>
                     </div>
                   )}

                   {/* Quick mode: 4-step indicator */}
                   {!completeCalibrationMode && calStep !== CalibrationStep.CREEP && (
                     <div className="bg-gray-800/50 rounded-xl p-3 mb-5">
                        <div className="flex items-center justify-center gap-0">
                          {[
                            { step: CalibrationStep.RELAX, label: '放松' },
                            { step: CalibrationStep.FIST, label: '握拳' },
                            { step: CalibrationStep.FLAT, label: '伸直' },
                            { step: CalibrationStep.SPREAD, label: '张开' },
                          ].map((item, idx, arr) => {
                            const stepOrder = [CalibrationStep.RELAX, CalibrationStep.FIST, CalibrationStep.FLAT, CalibrationStep.SPREAD];
                            const currentIdx = stepOrder.indexOf(calStep as any);
                            const thisIdx = idx;
                            const isPast = currentIdx > thisIdx;
                            const isCurrent = calStep === item.step;
                            return (
                              <React.Fragment key={idx}>
                                <div className="flex flex-col items-center gap-1">
                                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-[9px] font-bold ${isCurrent ? 'bg-cyan-600 text-white ring-2 ring-cyan-400/50' : isPast ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-400'}`}>{idx + 1}</div>
                                  <span className={`text-[7px] ${isCurrent ? 'text-cyan-400' : 'text-gray-500'}`}>{item.label}</span>
                                </div>
                                {idx < arr.length - 1 && (
                                  <div className={`w-8 h-0.5 ${isPast ? 'bg-green-600' : 'bg-gray-700'}`}></div>
                                )}
                              </React.Fragment>
                            );
                          })}
                        </div>
                        <p className="text-[7px] text-gray-500 text-center mt-2">快速标定仅需 4 步，完成后即可使用</p>
                     </div>
                   )}

                   <div className="bg-gray-800/30 rounded-lg p-4 mb-5">
                     <div className="flex items-start gap-3">
                       <div className={`mt-0.5 w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 ${
                         calStep === CalibrationStep.RELAX ? 'bg-cyan-500' :
                         calStep === CalibrationStep.FIST ? 'bg-cyan-500' :
                         calStep === CalibrationStep.FLAT ? 'bg-cyan-500' :
                         calStep === CalibrationStep.SPREAD ? 'bg-cyan-500' :
                         calStep === CalibrationStep.CREEP ? 'bg-yellow-500' : 'bg-gray-600'
                       }`}>
                         <span className="text-[8px] font-bold text-white">{
                           calStep === CalibrationStep.RELAX ? '1' :
                           calStep === CalibrationStep.FIST ? '2' :
                           calStep === CalibrationStep.FLAT ? '3' :
                           calStep === CalibrationStep.SPREAD ? '4' : '5'
                         }</span>
                       </div>
                       <div className="text-left">
                         <h3 className="text-sm font-bold text-gray-200 mb-1">
                           {calStep === CalibrationStep.RELAX && '放松基线标定'}
                           {calStep === CalibrationStep.FIST && '握拳标定'}
                           {calStep === CalibrationStep.FLAT && '伸直标定'}
                           {calStep === CalibrationStep.SPREAD && '张开标定'}
                           {calStep === CalibrationStep.CREEP && creepSubStep === 'HOLD_LOAD' && '蠕变标定 — 加载保持'}
                           {calStep === CalibrationStep.CREEP && creepSubStep === 'HOLD_RELEASE' && '蠕变标定 — 卸载恢复'}
                           {calStep === CalibrationStep.CREEP && creepSubStep === 'CYCLE' && '蠕变标定 — 滞回循环'}
                         </h3>
                         <p className="text-xs text-gray-400 leading-relaxed">
                           {calStep === CalibrationStep.RELAX && '自然放松手部，不要刻意伸直或弯曲，让手处于最舒适的状态。系统将记录传感器的零位基线。'}
                           {calStep === CalibrationStep.FIST && '用力握紧拳头，将大拇指搭在食指和中指的远端指间关节上方。保持姿势稳定。'}
                           {calStep === CalibrationStep.FLAT && '伸直并并拢五指。请务必使大拇指与手掌保持在同一平面，并紧贴食指侧面（不要前伸或内扣）。'}
                           {calStep === CalibrationStep.SPREAD && '保持手掌平伸在同一平面，同时尽力向外张开五指，包括大拇指虎口及四指间的间隙。'}
                           {calStep === CalibrationStep.CREEP && creepSubStep === 'HOLD_LOAD' && '保持中等握力（约 50% 弯曲）完全静止约 10 秒，系统正在记录传感器加载后的粘弹性蠕变上升曲线。请勿抖动。'}
                           {calStep === CalibrationStep.CREEP && creepSubStep === 'HOLD_RELEASE' && '快速松开至完全放松状态，然后保持静止约 10 秒，系统正在记录卸载后的蠕变恢复下降曲线。'}
                           {calStep === CalibrationStep.CREEP && creepSubStep === 'CYCLE' && '在 3 秒内快速握拳-松开 3 次，用于标定滞回特性。完成后点击下方按钮。'}
                         </p>
                       </div>
                     </div>
                   </div>

                   {/* B. 最低帧数提示 + 蠕变进度 */}
                   {calStep === CalibrationStep.CREEP && (creepSubStep === 'HOLD_LOAD' || creepSubStep === 'HOLD_RELEASE') && (
                     <div className="mb-5">
                        <div className="flex items-center gap-2 mb-2">
                           {/* 子步骤指示器 */}
                           <div className={`flex-1 h-1.5 rounded-full ${creepSubStep === 'HOLD_LOAD' ? 'bg-yellow-500' : 'bg-green-600'}`} />
                           <div className={`flex-1 h-1.5 rounded-full ${creepSubStep === 'HOLD_RELEASE' ? 'bg-yellow-500' : 'bg-gray-700'}`} />
                           <div className="flex-1 h-1.5 rounded-full bg-gray-700" />
                        </div>
                        <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden border border-gray-700">
                           <div
                              className={`h-full transition-all duration-100 ${
                                creepSubStep === 'HOLD_LOAD'
                                  ? 'bg-gradient-to-r from-yellow-600 to-yellow-400'
                                  : 'bg-gradient-to-r from-green-600 to-green-400'
                              }`}
                              style={{ width: `${Math.min(100, (calBuffer.length / 400) * 100)}%` }}
                           />
                        </div>
                        <div className="flex justify-between mt-2">
                           <p className="text-[10px] text-gray-500 font-mono">
                              {creepSubStep === 'HOLD_LOAD' ? '加载采样' : '卸载采样'}: {calBuffer.length} / 400 帧
                           </p>
                           <p className="text-[10px] text-yellow-500/70">
                              约 {Math.max(0, Math.round((400 - calBuffer.length) / 40))} 秒
                           </p>
                        </div>
                     </div>
                   )}
                   {calStep !== CalibrationStep.CREEP && calBuffer.length < MIN_CAL_FRAMES && (
                     <div className="mb-5 text-center">
                        <p className="text-[10px] text-orange-400/70">请保持姿势稳定，系统正在采样... ({calBuffer.length} / {MIN_CAL_FRAMES})</p>
                     </div>
                   )}
                   {calStep === CalibrationStep.CREEP && creepSubStep === 'CYCLE' && (
                     <div className="mb-5 text-center">
                        <p className="text-[10px] text-yellow-400/70">已采样 {calBuffer.length} 帧 — 完成屈伸循环后点击下方按钮</p>
                     </div>
                   )}

                   <div className="flex gap-3">
                      <button onClick={() => { setCalStep(CalibrationStep.IDLE); setCreepSubStep('HOLD_LOAD'); setCompleteCalibrationMode(false); }} className="flex-1 px-4 py-2.5 rounded-lg bg-gray-800 text-gray-300 text-sm border border-gray-700 hover:bg-gray-700/50 transition-colors">取消</button>
                      <button
                        onClick={nextCalibrationStep}
                        disabled={(calStep !== CalibrationStep.CREEP && calBuffer.length < MIN_CAL_FRAMES) || (calStep === CalibrationStep.CREEP && (creepSubStep === 'HOLD_LOAD' || creepSubStep === 'HOLD_RELEASE') && calBuffer.length < 30)}
                        className={`flex-[2] px-6 py-2.5 rounded-lg text-white font-bold text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg ${
                          completeCalibrationMode
                            ? 'bg-gradient-to-r from-cyan-600 via-yellow-500 to-purple-500 hover:from-cyan-500 hover:via-yellow-400 hover:to-purple-400 shadow-cyan-500/20'
                            : 'bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 shadow-cyan-500/20'
                        }`}
                      >
                        {calStep === CalibrationStep.CREEP && creepSubStep === 'HOLD_LOAD'
                          ? '下一步 — 松手保持'
                          : (calStep === CalibrationStep.CREEP && creepSubStep === 'HOLD_RELEASE'
                            ? '下一步 — 滞回循环'
                            : (calStep === CalibrationStep.CREEP && creepSubStep === 'CYCLE'
                              ? (completeCalibrationMode ? '完成蠕变标定 → 训练' : '完成蠕变标定')
                              : (calStep === CalibrationStep.SPREAD && completeCalibrationMode
                                ? '完成标定 → 蠕变标定'
                                : (calStep === CalibrationStep.SPREAD
                                  ? '完成标定'
                                  : '下一步'))))}
                      </button>
                   </div>
                </div>
             </div>
           )}
        </div>

        {showPanel && (
          <div className="w-80 md:w-96 bg-gray-900 border-l border-gray-800 flex flex-col h-full z-20 shadow-2xl overflow-hidden">
            <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-gray-900/50">
               <span className="text-[10px] font-bold uppercase text-gray-500 flex items-center gap-2"><Sliders size={14} /> 控制模式</span>
               <div className="flex bg-gray-800 rounded-md p-1 border border-gray-700">
                  <button onClick={() => { setManualMode(false); setTrainingMode(false); }} className={`px-2 py-1 text-[10px] font-bold rounded ${!manualMode && !trainingMode ? 'bg-cyan-600 text-white' : 'text-gray-400'}`}>传感器</button>
                  <button onClick={() => { setManualMode(true); setTrainingMode(false); }} className={`px-2 py-1 text-[10px] font-bold rounded ${manualMode ? 'bg-purple-600 text-white' : 'text-gray-400'}`}>手动</button>
                  <button onClick={() => { setTrainingMode(true); setManualMode(false); }} className={`px-2 py-1 text-[10px] font-bold rounded ${trainingMode ? 'bg-purple-600 text-white' : 'text-gray-400'}`}>训练</button>
               </div>
            </div>

            {trainingMode ? (
              <div className="flex-1 overflow-y-auto">
              <TrainingMode
                onBack={() => setTrainingMode(false)}
                onWeightsApplied={() => setMlpWeightsLoaded(true)}
                connected={connected}
                rawData={rawData}
              />
              </div>
            ) : (
            <div className="flex-1 overflow-y-auto">
            {/* 标定引导 - 可折叠 */}
            <div className="border-b border-gray-800">
                <div
                    className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-800/40 transition-all select-none bg-gray-850/50 border-b border-gray-750/30 shadow-sm"
                    onClick={() => toggleSection('calibration')}
                >
                    <div className="flex items-center gap-3">
                        <div className="w-2 h-5 bg-cyan-500 rounded-full shadow-[0_0_8px_rgba(6,182,212,0.5)]"></div>
                        <span className="text-[12px] font-bold text-white uppercase tracking-wider">标定引导</span>
                    </div>
                    {isCollapsed('calibration') ? <ChevronRight size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
                </div>
                {!isCollapsed('calibration') && (
                <div className="px-4 pb-3 bg-gradient-to-br from-cyan-900/20 to-gray-900/50">
                <div className="relative mb-4">
                    <div className="absolute left-[11px] top-8 bottom-8 w-0.5 bg-gray-700"></div>
                    
                    <div className="space-y-3 pl-7">
                        <div className={`relative flex items-start gap-2 p-2 rounded-lg ${calibrations.isCalibrated ? 'bg-green-900/20 border border-green-800/30' : 'bg-gray-800/30'}`}>
                            <div className={`absolute -left-[29px] w-5 h-5 rounded-full border-2 flex items-center justify-center ${calibrations.isCalibrated ? 'bg-green-500 border-green-600' : 'bg-gray-800 border-gray-600'}`}>
                                {calibrations.isCalibrated && <CheckCircle size={10} className="text-white" />}
                            </div>
                            <div>
                                <span className={`text-[9px] font-bold ${calibrations.isCalibrated ? 'text-green-400' : 'text-gray-400'}`}>常规标定</span>
                                <p className="text-[7px] text-gray-500 mt-0.5">确定传感器工作范围（放松/握拳/伸直/张开）</p>
                            </div>
                        </div>

                        <div className={`relative flex items-start gap-2 p-2 rounded-lg ${creepCalibrated ? 'bg-yellow-900/20 border border-yellow-800/30' : 'bg-gray-800/30'}`}>
                            <div className={`absolute -left-[29px] w-5 h-5 rounded-full border-2 flex items-center justify-center ${creepCalibrated ? 'bg-yellow-500 border-yellow-600' : 'bg-gray-800 border-gray-600'}`}>
                                {creepCalibrated && <CheckCircle size={10} className="text-white" />}
                            </div>
                            <div>
                                <span className={`text-[9px] font-bold ${creepCalibrated ? 'text-yellow-400' : 'text-gray-400'}`}>蠕变标定</span>
                                <p className="text-[7px] text-gray-500 mt-0.5">拟合液金粘弹性蠕变特性，消除滞后</p>
                            </div>
                        </div>

                        <div className={`relative flex items-start gap-2 p-2 rounded-lg ${mlpWeightsLoaded ? 'bg-purple-900/20 border border-purple-800/30' : 'bg-gray-800/30'}`}>
                            <div className={`absolute -left-[29px] w-5 h-5 rounded-full border-2 flex items-center justify-center ${mlpWeightsLoaded ? 'bg-purple-500 border-purple-600' : 'bg-gray-800 border-gray-600'}`}>
                                {mlpWeightsLoaded && <CheckCircle size={10} className="text-white" />}
                            </div>
                            <div>
                                <span className={`text-[9px] font-bold ${mlpWeightsLoaded ? 'text-purple-400' : 'text-gray-400'}`}>MLP解耦训练</span>
                                <p className="text-[7px] text-gray-500 mt-0.5">消除传感器间耦合，提升单指识别精度</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-gray-800/50 rounded-lg p-2 mb-3">
                    <div className="flex items-center gap-1 mb-2">
                        <Info size={10} className="text-cyan-400" />
                        <span className="text-[7px] text-gray-400">选择标定模式</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                        <button
                            onClick={() => startCalibration()}
                            disabled={calStep !== CalibrationStep.IDLE}
                            className="flex-1 px-3 py-2 rounded-lg bg-cyan-600/10 text-cyan-400 text-[9px] font-bold border border-cyan-600/20 hover:bg-cyan-600/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                            <span className="flex items-center justify-center gap-1">
                                <Sliders size={10} /> 快速标定
                            </span>
                        </button>
                        <button
                            onClick={() => startCalibration(true)}
                            disabled={calStep !== CalibrationStep.IDLE}
                            className="flex-1 px-3 py-2 rounded-lg bg-gradient-to-r from-cyan-600/20 via-yellow-600/20 to-purple-600/20 text-white text-[9px] font-bold border border-cyan-600/30 hover:from-cyan-600/30 hover:via-yellow-600/30 hover:to-purple-600/30 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                            <span className="flex items-center justify-center gap-1">
                                <Zap size={10} /> 完整标定
                            </span>
                        </button>
                    </div>
                </div>

                <div className="flex gap-2">
                    <button
                        onClick={exportCalParams}
                        disabled={!calibrations.isCalibrated}
                        className="flex-1 flex items-center justify-center gap-1 text-[8px] uppercase font-bold text-gray-400 hover:text-gray-200 transition-colors bg-gray-800/50 px-2 py-1.5 rounded border border-gray-700/50 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                        <Download size={10} /> 标定导出
                    </button>
                    <button
                        onClick={() => calFileInputRef.current?.click()}
                        className="flex-1 flex items-center justify-center gap-1 text-[8px] uppercase font-bold text-gray-400 hover:text-gray-200 transition-colors bg-gray-800/50 px-2 py-1.5 rounded border border-gray-700/50"
                    >
                        <Upload size={10} /> 标定导入
                    </button>
                    <input
                        ref={calFileInputRef}
                        type="file"
                        accept="application/json,.json"
                        onChange={importCalParams}
                        className="hidden"
                    />
                </div>

                <div className={`mt-3 p-2 rounded-lg border ${calibrations.isCalibrated && creepCalibrated && mlpWeightsLoaded ? 'bg-green-900/10 border-green-800/30' : calibrations.isCalibrated ? 'bg-yellow-900/10 border-yellow-800/30' : 'bg-red-900/10 border-red-800/30'}`}>
                    <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${calibrations.isCalibrated && creepCalibrated && mlpWeightsLoaded ? 'bg-green-500' : calibrations.isCalibrated ? 'bg-yellow-500' : 'bg-red-500'}`} />
                        <span className={`text-[7px] font-bold ${calibrations.isCalibrated && creepCalibrated && mlpWeightsLoaded ? 'text-green-400' : calibrations.isCalibrated ? 'text-yellow-400' : 'text-red-400'}`}>
                            {calibrations.isCalibrated && creepCalibrated && mlpWeightsLoaded ? '✓ 完整标定完成，精度最优' :
                             calibrations.isCalibrated ? '⚡ 快速标定完成，建议继续完整标定以提升精度' : '⚠️ 未标定，请先进行标定'}
                        </span>
                    </div>
                </div>
                </div>
                )}
            </div>

            {/* 数据录制 - 可折叠 */}
            <div className="border-b border-gray-800">
                <div
                    className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-800/40 transition-all select-none bg-gray-850/50 border-b border-gray-750/30 shadow-sm"
                    onClick={() => toggleSection('recording')}
                >
                    <div className="flex items-center gap-3">
                        <div className={`w-2 h-5 rounded-full shadow-sm ${isRecording ? 'bg-red-500 animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.5)]' : 'bg-gray-600'}`} />
                        <span className="text-[12px] font-bold text-white uppercase tracking-wider">数据录制</span>
                    </div>
                    {isCollapsed('recording') ? <ChevronRight size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
                </div>
                {!isCollapsed('recording') && (
                <div className="px-4 pb-3 bg-gray-800/50">
                <div className="flex gap-2">
                    <button
                        onClick={isRecording ? stopRecording : startRecording}
                        disabled={!connected}
                        className={`flex-1 flex items-center justify-center gap-1 text-[8px] uppercase font-bold px-2 py-1.5 rounded border transition-colors ${isRecording ? 'bg-red-600/20 text-red-400 border-red-600/40 hover:bg-red-600/30' : 'bg-green-600/10 text-green-400 border-green-600/20 hover:bg-green-600/20'} disabled:opacity-40 disabled:cursor-not-allowed`}
                    >
                        <RefreshCw size={10} className={isRecording ? 'animate-spin' : ''} />
                        {isRecording ? '停止录制' : '开始录制'}
                    </button>
                    <span className={`text-[8px] font-mono ${isRecording ? 'text-red-400' : 'text-gray-600'}`}>
                        {isRecording ? `${recordingBufferRef.current.length} frames` : '40Hz CSV'}
                    </span>
                </div>
                <p className="text-[7px] text-gray-600 mt-1.5">导出原始ADC数据为CSV，用于离线分析或Python训练</p>
                </div>
                )}
            </div>

            {/* CARD 算法控制 - 可折叠 */}
            <div className="border-b border-gray-800">
                <div
                    className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-800/40 transition-all select-none bg-gray-850/50 border-b border-gray-750/30 shadow-sm"
                    onClick={() => toggleSection('card')}
                >
                    <div className="flex items-center gap-3">
                        <Waves size={16} className="text-indigo-400" />
                        <span className="text-[12px] font-bold text-white uppercase tracking-wider">CARD 算法控制</span>
                    </div>
                    {isCollapsed('card') ? <ChevronRight size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
                </div>
                {!isCollapsed('card') && (
                <div className="px-4 pb-3 bg-indigo-500/5">
               <div className="flex justify-end mb-1">
                  <span className="text-[8px] text-indigo-500/60 font-mono">Stage 1-4 Pipeline</span>
               </div>
               <div className="grid grid-cols-2 gap-3">
                  <div className="flex items-center justify-between bg-gray-900/50 rounded-lg p-2 border border-gray-800">
                     <div className="flex items-center gap-2">
                        <Zap size={12} className={creepEnabled ? 'text-yellow-400' : 'text-gray-600'} />
                        <span className="text-[9px] text-gray-300">蠕变补偿</span>
                     </div>
                     <button 
                        onClick={() => setCreepEnabled(!creepEnabled)}
                        className={`w-8 h-4 rounded-full transition-colors ${creepEnabled ? 'bg-yellow-500' : 'bg-gray-700'}`}
                     >
                        <div className={`w-3 h-3 rounded-full bg-white transition-transform ${creepEnabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
                     </button>
                  </div>
                  <div className="flex items-center justify-between bg-gray-900/50 rounded-lg p-2 border border-gray-800">
                     <div className="flex items-center gap-2">
                        <Brain size={12} className={decouplerEnabled ? 'text-purple-400' : 'text-gray-600'} />
                        <span className="text-[9px] text-gray-300">MLP 解耦</span>
                     </div>
                     <button 
                        onClick={() => setDecouplerEnabled(!decouplerEnabled)}
                        className={`w-8 h-4 rounded-full transition-colors ${decouplerEnabled ? 'bg-purple-500' : 'bg-gray-700'}`}
                     >
                        <div className={`w-3 h-3 rounded-full bg-white transition-transform ${decouplerEnabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
                     </button>
                  </div>
               </div>
               <div className="mt-3">
                  <div className="flex justify-between text-[9px] mb-1">
                     <span className="text-gray-400">平滑系数 (α)</span>
                     <span className="font-mono text-indigo-400">{smoothingAlpha.toFixed(2)}</span>
                  </div>
                  <input
                     type="range"
                     min="0.1"
                     max="0.5"
                     step="0.01"
                     value={smoothingAlpha}
                     onChange={(e) => setSmoothingAlpha(parseFloat(e.target.value))}
                     className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
               </div>
               <div className="mt-3 pt-3 border-t border-gray-800/60">
                  <div className="flex items-center justify-between mb-2">
                     <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${creepCalibrated ? 'bg-green-500' : 'bg-orange-500'}`} />
                        <span className="text-[9px] text-gray-400 uppercase font-bold">
                           {creepCalibrated ? '蠕变参数已标定' : '蠕变参数未标定'}
                        </span>
                     </div>
                     <button
                        onClick={startCreepCalibration}
                        disabled={calStep !== CalibrationStep.IDLE}
                        className="text-[8px] uppercase font-bold text-yellow-400 hover:text-yellow-300 transition-colors bg-yellow-400/10 px-2 py-1 rounded border border-yellow-400/20 disabled:opacity-40 disabled:cursor-not-allowed"
                     >
                        蠕变标定
                     </button>
                  </div>
                  <div className="flex gap-2">
                     <button
                        onClick={exportCreepParams}
                        disabled={!creepCalibrated}
                        className="flex-1 flex items-center justify-center gap-1 text-[8px] uppercase font-bold text-gray-400 hover:text-gray-200 transition-colors bg-gray-800/50 px-2 py-1.5 rounded border border-gray-700/50 disabled:opacity-40 disabled:cursor-not-allowed"
                     >
                        <Download size={10} /> 蠕变导出
                     </button>
                     <button
                        onClick={() => creepFileInputRef.current?.click()}
                        className="flex-1 flex items-center justify-center gap-1 text-[8px] uppercase font-bold text-gray-400 hover:text-gray-200 transition-colors bg-gray-800/50 px-2 py-1.5 rounded border border-gray-700/50"
                     >
                        <Upload size={10} /> 蠕变导入
                     </button>
                     <input
                        ref={creepFileInputRef}
                        type="file"
                        accept="application/json,.json"
                        onChange={importCreepParams}
                        className="hidden"
                     />
                  </div>
               </div>
               <div className="mt-3 pt-3 border-t border-gray-800/60">
                  <div className="flex items-center justify-between mb-2">
                     <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${mlpWeightsLoaded ? 'bg-green-500' : 'bg-orange-500'}`} />
                        <span className="text-[9px] text-gray-400 uppercase font-bold">
                           {mlpWeightsLoaded ? 'MLP权重已加载' : 'MLP权重未加载'}
                        </span>
                     </div>
                     <button
                        onClick={() => mlpWeightsFileRef.current?.click()}
                        className="text-[8px] uppercase font-bold text-purple-400 hover:text-purple-300 transition-colors bg-purple-400/10 px-2 py-1 rounded border border-purple-400/20"
                     >
                        <Upload size={10} className="inline mr-1" />导入权重
                     </button>
                  </div>
                  <input
                     ref={mlpWeightsFileRef}
                     type="file"
                     accept="application/json,.json"
                     onChange={importMLPWeights}
                     className="hidden"
                  />
               </div>
               </div>
               )}
            </div>

            {/* 实时数据 - 可折叠，包含各手指数据 */}
            <div>
                <div
                    className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-800/40 transition-all select-none sticky top-0 bg-gray-850/90 backdrop-blur-md z-10 border-b border-gray-750/30 shadow-sm"
                    onClick={() => toggleSection('sensors')}
                >
                    <div className="flex items-center gap-3">
                        <Activity size={16} className="text-cyan-400" />
                        <span className="text-[12px] font-bold text-white uppercase tracking-wider">实时数据</span>
                    </div>
                    {isCollapsed('sensors') ? <ChevronRight size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
                </div>
                {!isCollapsed('sensors') && (
                <div className="p-4 space-y-4">
               {/* 1. THUMB */}
               <div className="bg-gray-800/20 rounded-xl border border-gray-800 overflow-hidden">
                  <div className="px-4 py-2 bg-gray-800/50 border-b border-gray-800 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                    <h3 className="text-[10px] font-bold text-gray-300 uppercase">大拇指 Thumb</h3>
                  </div>
                  <div className="p-3">
                    {renderSensorControl("腕掌关节内扣 (CMC)", SENSOR_MAP.THUMB_MCP)}
                    {renderWeightedDisplay("加权掌指关节 (Calc MCP)", cardResult.derived.thumbWeightedMCP as number)}
                    {renderSensorControl("指间关节 (IP)", SENSOR_MAP.THUMB_IP)}
                    {renderSensorControl("腕掌关节张开 (Spread)", SENSOR_MAP.THUMB_SPREAD)}
                  </div>
               </div>

               {/* 2. INDEX */}
               <div className="bg-gray-800/20 rounded-xl border border-gray-800 overflow-hidden">
                  <div className="px-4 py-2 bg-gray-800/50 border-b border-gray-800">
                    <h3 className="text-[10px] font-bold text-gray-300 uppercase">食指 Index</h3>
                  </div>
                  <div className="p-3">
                    {renderSensorControl("掌指关节 (MCP)", SENSOR_MAP.INDEX_MCP)}
                    {renderWeightedDisplay("加权掌指关节 (Calc MCP)", cardResult.derived.indexWeightedMCP as number)}
                    {renderSensorControl("近端指间关节 (PIP)", SENSOR_MAP.INDEX_PIP)}
                    {renderSensorControl("食中张开 (Spread)", SENSOR_MAP.INDEX_MIDDLE_SPREAD)}
                    {renderDIPControl("远端指间关节 (DIP)", "index")}
                  </div>
               </div>

               {/* 3. MIDDLE */}
               <div className="bg-gray-800/20 rounded-xl border border-gray-800 overflow-hidden">
                  <div className="px-4 py-2 bg-gray-800/50 border-b border-gray-800">
                    <h3 className="text-[10px] font-bold text-gray-300 uppercase">中指 Middle</h3>
                  </div>
                  <div className="p-3">
                    {renderSensorControl("掌指关节 (MCP)", SENSOR_MAP.MIDDLE_MCP)}
                    {renderWeightedDisplay("加权掌指关节 (Calc MCP)", cardResult.derived.middleWeightedMCP as number)}
                    {renderSensorControl("近端指间关节 (PIP)", SENSOR_MAP.MIDDLE_PIP)}
                    {renderSensorControl("中无张开 (Spread)", SENSOR_MAP.MIDDLE_RING_SPREAD)}
                    {renderDIPControl("远端指间关节 (DIP)", "middle")}
                  </div>
               </div>

               {/* 4. RING */}
               <div className="bg-gray-800/20 rounded-xl border border-gray-800 overflow-hidden">
                  <div className="px-4 py-2 bg-gray-800/50 border-b border-gray-800">
                    <h3 className="text-[10px] font-bold text-gray-300 uppercase">无名指 Ring</h3>
                  </div>
                  <div className="p-3">
                    {renderSensorControl("指关节 (Flexion)", SENSOR_MAP.RING_PIP)}
                    {renderSensorControl("无小张开 (Spread)", SENSOR_MAP.RING_PINKY_SPREAD)}
                    {renderDIPControl("远端指间关节 (DIP)", "ring")}
                  </div>
               </div>

               {/* 5. PINKY */}
               <div className="bg-gray-800/20 rounded-xl border border-gray-800 overflow-hidden">
                  <div className="px-4 py-2 bg-gray-800/50 border-b border-gray-800">
                    <h3 className="text-[10px] font-bold text-gray-300 uppercase">小指 Pinky</h3>
                  </div>
                  <div className="p-3">
                    {renderSensorControl("指关节 (Flexion)", SENSOR_MAP.PINKY_PIP)}
                    {renderDIPControl("远端指间关节 (DIP)", "pinky")}
                  </div>
               </div>
                </div>
                )}
            </div>
            
            <div className="px-4 py-2 border-t border-gray-800 bg-gray-950 text-[9px] text-gray-600 flex justify-between items-center">
                <span className="flex items-center gap-1"><RefreshCw size={10} /> 12ch @ 460800 baud</span>
                <span className="font-mono text-cyan-800 uppercase">RIGHT HAND MONITOR ACTIVE</span>
            </div>
            </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
