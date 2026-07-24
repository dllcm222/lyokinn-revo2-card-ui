
import { HandSide } from '../types';

interface SerialPort {
  readable: ReadableStream<Uint8Array> | null;
  writable: WritableStream<Uint8Array> | null;
  open(options: { baudRate: number }): Promise<void>;
  close(): Promise<void>;
  getInfo(): { usbVendorId?: number; usbProductId?: number };
}

declare global {
  interface Navigator {
    serial: {
      requestPort(options?: { filters?: Array<{ usbVendorId?: number; usbProductId?: number }> }): Promise<SerialPort>;
      getPorts(): Promise<SerialPort[]>;
      addEventListener(type: string, listener: (event: Event) => void): void;
      removeEventListener(type: string, listener: (event: Event) => void): void;
    };
  }
}

export class SerialService {
  private ports: Map<HandSide, SerialPort> = new Map();
  private readers: Map<HandSide, ReadableStreamDefaultReader<Uint8Array>> = new Map();
  private callbacks: Map<HandSide, (data: number[]) => void> = new Map();

  constructor() {}

  async getAuthorizedPorts(): Promise<SerialPort[]> {
    if (!navigator.serial) return [];
    return await navigator.serial.getPorts();
  }

  async requestAccess(): Promise<SerialPort | null> {
    if (!navigator.serial) throw new Error('Web Serial API not supported');
    try {
      return await navigator.serial.requestPort();
    } catch (error) {
      console.log('User cancelled port request', error);
      return null;
    }
  }

  async connectToPort(side: HandSide, port: SerialPort, onData: (data: number[]) => void): Promise<boolean> {
    if (!port) return false;
    
    // Check if this specific port instance is already open by checking its readable attribute
    // or if it's already registered to any side in our service
    const isPortUsedByOtherSide = Array.from(this.ports.entries()).some(([s, p]) => s !== side && p === port);
    if (isPortUsedByOtherSide) {
      console.warn(`Port is already in use by another hand side.`);
      return false;
    }

    try {
      // If we're already connected to SOMETHING on this side, disconnect first
      if (this.ports.has(side)) {
        await this.disconnect(side);
      }

      // Try to open. If it's already open, it might throw or we might handle it.
      try {
        await port.open({ baudRate: 460800 });
      } catch (innerError: any) {
        // If the error is specifically that it's already open, we might be able to proceed 
        // if we are the ones who own it, but usually this means another tab or process has it.
        if (innerError.message?.includes('already open')) {
          console.log('Port already open, attempting to use existing connection');
          // If we can't be sure, we still proceed to set up our listeners
        } else {
          throw innerError;
        }
      }

      this.ports.set(side, port);
      this.callbacks.set(side, onData);
      this.readLoop(side);
      return true;
    } catch (error) {
      console.error(`Serial connection failed for ${side}:`, error);
      return false;
    }
  }

  async disconnect(side: HandSide) {
    const reader = this.readers.get(side);
    if (reader) {
      try {
        await reader.cancel();
      } catch (e) {}
      this.readers.delete(side);
    }
    const port = this.ports.get(side);
    if (port) {
      try {
        await port.close();
      } catch (e) {}
      this.ports.delete(side);
    }
    this.callbacks.delete(side);
  }

  private async readLoop(side: HandSide) {
    const port = this.ports.get(side);
    if (!port || !port.readable) return;

    const reader = port.readable.getReader();
    this.readers.set(side, reader);

    const FRAME_HEAD = [0xAA, 0x55];
    const CHANNEL_NUM = 16;
    const FRAME_LEN = 38; // 2 + 1 + 2 + 2 * 16 + 1
    
    let buffer = new Uint8Array(0);

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) {
          // Append new data to buffer
          const newBuffer = new Uint8Array(buffer.length + value.length);
          newBuffer.set(buffer);
          newBuffer.set(value, buffer.length);
          buffer = newBuffer;

          // Process buffer
          while (buffer.length >= FRAME_LEN) {
            // Find header
            let headIdx = -1;
            for (let i = 0; i <= buffer.length - FRAME_LEN; i++) {
              if (buffer[i] === FRAME_HEAD[0] && buffer[i + 1] === FRAME_HEAD[1]) {
                headIdx = i;
                break;
              }
            }

            if (headIdx === -1) {
              // No header found, keep only the last byte if it might be 0xAA
              if (buffer[buffer.length - 1] === FRAME_HEAD[0]) {
                buffer = buffer.slice(buffer.length - 1);
              } else {
                buffer = new Uint8Array(0);
              }
              break;
            }

            // Sync to header
            if (headIdx > 0) {
              buffer = buffer.slice(headIdx);
            }

            if (buffer.length < FRAME_LEN) break;

            const frame = buffer.slice(0, FRAME_LEN);
            const parsed = this.parseFrame(frame, CHANNEL_NUM);

            if (parsed) {
              const callback = this.callbacks.get(side);
              // Extract first 12 channels as the app expects 12 sensors
              if (callback) callback(parsed.raws.slice(0, 12));
              buffer = buffer.slice(FRAME_LEN);
            } else {
              // Invalid frame, skip 1 byte and try again
              buffer = buffer.slice(1);
            }
          }
        }
      }
    } catch (error) {
      console.error(`Read error for ${side}:`, error);
    } finally {
      reader.releaseLock();
    }
  }

  private parseFrame(buf: Uint8Array, channelNum: number) {
    if (buf[2] !== channelNum) return null;

    // XOR Checksum
    let chk = 0;
    for (let i = 2; i < buf.length - 1; i++) {
      chk ^= buf[i];
    }
    if (chk !== buf[buf.length - 1]) return null;

    // const seq = buf[3] | (buf[4] << 8); // Sequence number (2 bytes little-endian)
    const raws: number[] = [];
    for (let i = 0; i < channelNum; i++) {
      const off = 5 + i * 2;
      raws.push(buf[off] | (buf[off + 1] << 8));
    }
    return { raws };
  }
}

export const serialService = new SerialService();
