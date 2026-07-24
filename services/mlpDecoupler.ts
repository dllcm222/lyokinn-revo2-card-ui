import { MLPWeights, DecouplerState } from '../types';

function randomInit(rows: number, cols: number): number[][] {
  const weights: number[][] = [];
  const scale = Math.sqrt(2.0 / cols);
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push((Math.random() - 0.5) * 2 * scale);
    }
    weights.push(row);
  }
  return weights;
}

function relu(x: number): number {
  return Math.max(0, x);
}

function tanh(x: number): number {
  return Math.tanh(x);
}

function leakyRelu(x: number, alpha: number = 0.01): number {
  return x > 0 ? x : alpha * x;
}

function createDefaultWeights(): MLPWeights {
  return {
    input: {
      weights: randomInit(32, 24),
      biases: Array(32).fill(0),
    },
    hidden: {
      weights: randomInit(32, 32),
      biases: Array(32).fill(0),
    },
    output: {
      weights: randomInit(12, 32),
      biases: Array(12).fill(0),
    },
  };
}

export class MLPDecoupler {
  private weights: MLPWeights;
  private state: DecouplerState;
  private enabled: boolean;

  constructor() {
    this.weights = createDefaultWeights();
    this.state = {
      previousOutput: Array(12).fill(0),
    };
    this.enabled = true;
  }

  setWeights(weights: MLPWeights) {
    this.weights = weights;
  }

  getWeights(): MLPWeights {
    return JSON.parse(JSON.stringify(this.weights)) as MLPWeights;
  }

  setEnabled(enabled: boolean) {
    this.enabled = enabled;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  reset() {
    this.state.previousOutput = Array(12).fill(0);
  }

  private forward(input: number[]): number[] {
    const { input: inputLayer, hidden: hiddenLayer, output: outputLayer } = this.weights;

    let hidden1: number[] = [];
    for (let i = 0; i < inputLayer.weights.length; i++) {
      let sum = inputLayer.biases[i];
      for (let j = 0; j < input.length; j++) {
        sum += inputLayer.weights[i][j] * input[j];
      }
      hidden1.push(leakyRelu(sum));
    }

    let hidden2: number[] = [];
    for (let i = 0; i < hiddenLayer.weights.length; i++) {
      let sum = hiddenLayer.biases[i];
      for (let j = 0; j < hidden1.length; j++) {
        sum += hiddenLayer.weights[i][j] * hidden1[j];
      }
      hidden2.push(leakyRelu(sum));
    }

    let output: number[] = [];
    for (let i = 0; i < outputLayer.weights.length; i++) {
      let sum = outputLayer.biases[i];
      for (let j = 0; j < hidden2.length; j++) {
        sum += outputLayer.weights[i][j] * hidden2[j];
      }
      output.push(tanh(sum));
    }

    return output;
  }

  process(compensated: number[]): number[] {
    if (!this.enabled) {
      this.state.previousOutput = compensated.slice();
      return compensated;
    }

    const input = [...compensated, ...this.state.previousOutput];
    const output = this.forward(input);

    this.state.previousOutput = output.slice();

    return output;
  }

  getState(): DecouplerState {
    return {
      previousOutput: [...this.state.previousOutput],
    };
  }

  exportWeights(): string {
    return JSON.stringify(this.weights);
  }

  importWeights(jsonStr: string): boolean {
    try {
      const weights = JSON.parse(jsonStr) as MLPWeights;
      this.setWeights(weights);
      return true;
    } catch {
      return false;
    }
  }
}

export const mlpDecoupler = new MLPDecoupler();