import { HandSide } from '../types';

// Modbus RTU CRC16 Calculator
export function calculateCRC16(buffer: Uint8Array): number {
  let crc = 0xFFFF;
  for (let pos = 0; pos < buffer.length; pos++) {
    crc ^= buffer[pos];
    for (let i = 8; i !== 0; i--) {
      if ((crc & 0x0001) !== 0) {
        crc >>= 1;
        crc ^= 0xA001;
      } else {
        crc >>= 1;
      }
    }
  }
  return crc;
}

export class Revo2Service {
  private port: any | null = null;
  private writer: WritableStreamDefaultWriter<Uint8Array> | null = null;
  private isWriting = false;

  // WebSocket for Local Python Bridge (SDK)
  private ws: WebSocket | null = null;
  private wsConnected = false;

  constructor() {}

  // ==========================================
  // Connection 1: Local Python Bridge (SDK)
  // ==========================================
  async connectWebSocket(): Promise<boolean> {
    return new Promise((resolve) => {
      try {
        this.ws = new WebSocket('ws://localhost:8765');
        
        this.ws.onopen = () => {
          this.wsConnected = true;
          resolve(true);
        };
        
        this.ws.onerror = (e) => {
          console.error("WebSocket Error, make sure Python bridge is running at ws://localhost:8765");
          this.wsConnected = false;
          resolve(false);
        };
        
        this.ws.onclose = () => {
          this.wsConnected = false;
        };
      } catch (e) {
        resolve(false);
      }
    });
  }

  isWsConnected() {
    return this.wsConnected;
  }

  disconnectWs() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.wsConnected = false;
  }

  sendFingerPositionsWs(
    thumbFlexion: number, 
    thumbRotation: number, 
    index: number, 
    middle: number, 
    ring: number, 
    pinky: number
  ) {
    if (!this.wsConnected || !this.ws) return;
    
    // Scale 0~1.0 to Revo2 expected range (0=open, 1000=closed).
    const scale = (val: number) => Math.round(Math.max(0, Math.min(1, val)) * 1000);
    const values = [
      scale(thumbFlexion),  // Thumb
      scale(thumbRotation), // ThumbAux (Rotation)
      scale(index),         // Index
      scale(middle),        // Middle
      scale(ring),          // Ring
      scale(pinky)          // Pinky
    ];

    this.ws.send(JSON.stringify({ type: 'positions', data: values }));
  }


  // ==========================================
  // Connection 2: Raw Web Serial (Direct)
  // ==========================================
  async requestAndConnect(): Promise<boolean> {
    if (!navigator.serial) {
      console.error('Web Serial API not supported');
      return false;
    }
    
    try {
      let selectedPort: any;
      try {
        selectedPort = await navigator.serial.requestPort();
      } catch (err: any) {
        if (err.name === 'NotFoundError' || err.message?.includes('No port selected')) {
          return false; // User cancelled
        }
        throw err;
      }
      
      if (!selectedPort) return false;
      
      await selectedPort.open({ baudRate: 460800 });
      this.port = selectedPort;
      this.writer = this.port.writable.getWriter();
      return true;
    } catch (e: any) {
      if (e.name === 'InvalidStateError' || e.message?.includes('already open')) {
        console.warn('Port already open. Another device or service might be using it.');
        return false;
      }
      if (e.name === 'NetworkError' || e.message?.includes('Failed to open serial port')) {
        console.warn('Failed to open serial port. It may be in use by another application.');
        return false;
      }
      console.error('Revo2 connection error:', e);
      return false;
    }
  }

  async disconnect() {
    if (this.writer) {
      await this.writer.close().catch(() => {});
      this.writer.releaseLock();
      this.writer = null;
    }
    if (this.port) {
      await this.port.close().catch(() => {});
      this.port = null;
    }
  }

  isConnected() {
    return !!this.port && !!this.writer;
  }

  /**
   * 发送多寄存器写入命令 (Function Code 16 / 0x10)
   * 默认ID是127 (0x7F)
   */
  async writeMultipleRegisters(slaveId: number, startAddress: number, values: number[]) {
    if (!this.writer || this.isWriting) return;
    this.isWriting = true;

    try {
      const numRegisters = values.length;
      const byteCount = numRegisters * 2;
      
      // MB RTU Frame: 
      // SlaveAddress(1) + Func(1) + StartAddr(2) + NumRegs(2) + ByteCount(1) + Data(N) + CRC(2)
      const frameLength = 7 + byteCount + 2; 
      const frame = new Uint8Array(frameLength);
      
      frame[0] = slaveId;
      frame[1] = 0x10; // Function 16
      frame[2] = (startAddress >> 8) & 0xFF; // Hi
      frame[3] = startAddress & 0xFF; // Lo
      frame[4] = (numRegisters >> 8) & 0xFF;
      frame[5] = numRegisters & 0xFF;
      frame[6] = byteCount;
      
      let offset = 7;
      for (const val of values) {
        frame[offset++] = (val >> 8) & 0xFF;
        frame[offset++] = val & 0xFF;
      }
      
      const crc = calculateCRC16(frame.slice(0, frameLength - 2));
      frame[frameLength - 2] = crc & 0xFF; // Lo
      frame[frameLength - 1] = (crc >> 8) & 0xFF; // Hi
      
      await this.writer.write(frame);
    } catch (e) {
      console.error('Error writing to Revo2:', e);
    } finally {
      this.isWriting = false;
    }
  }

  /**
   * 基于传入的各个手指归一化角度(0.0~1.0)，转换为Revo2需要的数值范围发送。
   * 参考Revo2协议，我们将此处寄存器地址设置为可配置（如果官网有特定地址，可在此修改）
   * 常见BrainCo Revo2 控制地址例如 0x0001等，手指顺序大拇指等。
   */
  async sendFingerPositions(
    thumbFlexion: number, 
    thumbRotation: number, 
    index: number, 
    middle: number, 
    ring: number, 
    pinky: number,
    startAddress: number = 0x0110
  ) {
    if (!this.isConnected()) return;
    
    // Scale 0~1.0 to Revo2 expected range.
    // 0 = fully open, 1000 = fully closed
    const scale = (val: number) => Math.round(Math.max(0, Math.min(1, val)) * 1000);
    
    // Order according to libstark: [Thumb, ThumbAux, Index, Middle, Ring, Pinky]
    const values = [
      scale(thumbFlexion),  // Thumb
      scale(thumbRotation), // ThumbAux (Rotation)
      scale(index),         // Index
      scale(middle),        // Middle
      scale(ring),          // Ring
      scale(pinky)          // Pinky
    ];

    // 发送到 127 默认，基于UI传入的具体寄存器地址
    // 假设连续写入6个保持寄存器
    await this.writeMultipleRegisters(127, startAddress, values); 
  }
}

export const revo2Service = new Revo2Service();
