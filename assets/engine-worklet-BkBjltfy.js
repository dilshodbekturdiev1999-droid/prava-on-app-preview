/*
 * Engine sound for the avtodrom: a physically informed piston-engine model
 * (cylinders with valves, intake / exhaust / extractor waveguides, straight pipe,
 * muffler, outlet), after Baldan et al. 2015, "Physically informed car engine
 * sound synthesis for virtual and augmented environments".
 *
 * Adapted from engine-sound-generator by Antonio-R1
 *   Copyright (c) 2021-2022 Antonio-R1, MIT License
 *   https://github.com/Antonio-R1/engine-sound-generator/blob/main/LICENSE
 * Changes: engine load (throttle) drives the combustion strength and intake
 * noise, firing jitter per cycle, diesel injector knock, one mixed output,
 * sample-rate independent filters.
 */

class Lowpass {
  constructor(freq, rate) {
    const w = (2 * Math.PI * freq) / rate;
    this.a = w / (w + 1);
    this.v = 0;
  }
  run(x) {
    this.v += this.a * (x - this.v);
    return this.v;
  }
}

class DelayLine {
  constructor(n) {
    this.d = new Float32Array(Math.max(1, n | 0));
    this.i = 0;
  }
  pushLeft(v) {
    this.d[this.i] = v;
    if (++this.i >= this.d.length) this.i = 0;
  }
  pushRight(v) {
    this.d[this.i] = v;
    if (--this.i < 0) this.i = this.d.length - 1;
  }
  get head() {
    return this.d[this.i];
  }
}

// a little energy is lost at every reflection (air friction, pipe walls); keeps the
// coupled pipes stable whatever lengths / reflection factors a preset uses
const LOSS = 0.985;

class Waveguide {
  constructor(n, rl, rr) {
    this.up = new DelayLine(n);
    this.lo = new DelayLine(n);
    this.rl = rl;
    this.rr = rr;
    this.outL = 0;
    this.outR = 0;
  }
  add(inL, inR) {
    const lo = this.lo.head;
    const up = this.up.head;
    this.outL = lo * (1 - this.rl);
    this.outR = up * (1 - this.rr);
    this.up.pushRight(inL + lo * this.rl * LOSS);
    this.lo.pushLeft(inR + up * this.rr * LOSS);
  }
}

class Cylinder {
  constructor(p) {
    this.cyl = new Waveguide(10, 0.75, 0.75);
    this.intake = new Waveguide(p.intake, 0.01, p.intakeOpen);
    this.exhaust = new Waveguide(p.exhaust, p.exhaustClosed, 0.01);
    this.extractor = new Waveguide(p.extractor, 0.01, 0.01);
    this.p = p;
    this.jitter = 1;
  }
  update(noise, pipeL, x, load) {
    const p = this.p;
    const ex = x > 0.75 ? -Math.sin(4 * Math.PI * x) : 0;
    const inV = x < 0.25 ? Math.sin(4 * Math.PI * x) : 0;
    const piston = Math.cos(4 * Math.PI * x);
    const t = p.ignition;
    const fire = x < 0.5 * t ? Math.sin((2 * Math.PI * x) / t) : 0;
    const rIn = p.intakeOpen * inV + p.intakeClosed * (1 - inV);
    const rEx = p.exhaustOpen * ex + p.exhaustClosed * (1 - ex);
    this.intake.rr = rIn;
    this.cyl.rl = rIn;
    this.exhaust.rl = rEx;
    this.cyl.rr = rEx;
    // combustion strength follows the load (part throttle is softer, overrun almost silent)
    const combustion = fire * 5 * (0.32 + 0.68 * load) * this.jitter;
    const amp = piston * 1.5 + combustion;
    const exL = this.extractor.outL;
    const cylL = this.cyl.outL;
    const cylR = this.cyl.outR;
    const inR = this.intake.outR;
    this.extractor.add(this.exhaust.outR, pipeL);
    this.exhaust.add(cylR, exL);
    this.cyl.add(amp + inR * (1 - this.intake.rr), exL * (1 - this.exhaust.rl));
    this.intake.add(noise * inV, cylL * (1 - this.intake.rr));
    return fire;
  }
}

class Muffler {
  constructor(lengths, action) {
    this.el = lengths.map((n) => new Waveguide(n, 0, action));
    this.k = 1 / lengths.length;
    this.outL = 0;
    this.outR = 0;
  }
  update(input, outlet) {
    const a = input * this.k;
    const b = outlet * this.k;
    let l = 0;
    let r = 0;
    for (const e of this.el) {
      l += e.outL;
      r += e.outR;
      e.add(a, b);
    }
    this.outL = l;
    this.outR = r;
  }
}

class Engine extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'rpm', defaultValue: 0, minValue: 0, automationRate: 'k-rate' },
      { name: 'load', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();
    const p = options.processorOptions;
    this.p = p;
    this.cyl = [];
    for (let i = 0; i < p.cylinders; i++) this.cyl.push(new Cylinder(p));
    this.pipe = new Waveguide(p.pipe, p.pipeReflect, p.pipeReflect);
    this.muffler = new Muffler(p.muffler, p.action);
    this.outlet = new Waveguide(p.outlet, 0.01, 0.01);
    this.noiseLp = new Lowpass(p.intakeNoiseHz, sampleRate);
    this.crankLp = new Lowpass(60, sampleRate);
    this.blockLp = new Lowpass(p.blockHz, sampleRate);
    this.knockLp = new Lowpass(2600, sampleRate);
    this.phase = 0;
    this.lastX = [];
    this.knock = 0;
    this.dc = 0;
  }

  process(_in, outputs, params) {
    const out = outputs[0][0];
    if (!out) return true;
    const rpm = params.rpm[0];
    const load = params.load[0];
    const p = this.p;
    const n = this.cyl.length;
    const inv = 1 / n;
    const step = rpm / 120 / sampleRate; // one cycle = two revolutions
    for (let s = 0; s < out.length; s++) {
      // intake noise: idleNoise share at closed throttle (presets without it keep 0.35)
      const idleNoise = p.idleNoise ?? 0.35;
      let noise = this.noiseLp.run(Math.random() * 2 - 1) * (idleNoise + (1 - idleNoise) * load);
      if (rpm < 25) noise = 0;
      // evenFire presets: tiny zero-mean crank jitter, cylinders fire exactly 1/n apart
      // (smooth petrol idle; `lope` adds a small fixed offset for a little character); the others keep their lumpy offset (the diesel's character)
      const crank = p.evenFire
        ? this.crankLp.run(Math.random() - 0.5) * 0.02 * p.roughness + (p.lope ?? 0)
        : this.crankLp.run(0.25 * Math.random()) * p.roughness;
      let block = 0;
      let intake = 0;
      let pipeIn = 0;
      for (let i = 0; i < n; i++) {
        const c = this.cyl[i];
        const x = (this.phase + i * (inv + crank)) % 1;
        // new firing cycle: a little random strength (real combustion is never identical)
        if (x < (this.lastX[i] ?? 1)) c.jitter = 1 - p.misfire * Math.random();
        this.lastX[i] = x;
        const fire = c.update(noise, this.pipe.outL, x, load);
        if (p.knock > 0 && fire > 0.95) this.knock = 1;
        block += c.cyl.outR;
        intake += c.intake.outL;
        pipeIn += c.extractor.outR;
      }
      this.phase += step;
      if (this.phase >= 1) this.phase -= 1;
      this.pipe.add(Math.tanh(pipeIn * 0.25) * 4, this.muffler.outL);
      this.outlet.add(this.muffler.outR, 0);
      this.muffler.update(this.pipe.outR, this.outlet.outL);
      // diesel: a short bright tick at each injection
      let knock = 0;
      if (this.knock > 0.001) {
        knock = this.knockLp.run((Math.random() * 2 - 1) * this.knock) * p.knock;
        this.knock *= 0.93;
      }
      const v = this.blockLp.run(block) * p.mixBlock + intake * p.mixIntake + this.outlet.outR * p.mixOutlet + knock;
      // remove DC drift
      this.dc += 0.004 * (v - this.dc);
      // soft limit: never a hard clip, whatever the revs
      const y = (v - this.dc) * p.trim;
      out[s] = y / (1 + Math.abs(y));
    }
    for (let ch = 1; ch < outputs[0].length; ch++) outputs[0][ch].set(out);
    return true;
  }
}

registerProcessor('avtodrom-engine', Engine);
