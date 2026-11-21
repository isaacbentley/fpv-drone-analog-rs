//! Realistic RF impairment models for weak-signal testing.

use num_complex::Complex;

/// Adds multipath fading (echoes) to the signal.
/// `echoes` is a list of (delay_samples, attenuation_factor, initial_phase_radians, doppler_hz).
pub fn add_multipath(iq: &mut [Complex<f32>], echoes: &[(usize, f32, f32, f32)], sample_rate: f32) {
    // We must work on a copy since echoes depend on older samples
    let original = iq.to_vec();
    for &(delay, atten, initial_phase, doppler_hz) in echoes {
        let omega = 2.0 * std::f32::consts::PI * doppler_hz / sample_rate;
        for i in delay..iq.len() {
            let current_phase = initial_phase + omega * i as f32;
            let p = Complex::new(current_phase.cos(), current_phase.sin());
            iq[i] += original[i - delay] * atten * p;
        }
    }
}

/// Clamps the signal to pure AWGN (burst dropout) for the specified runs.
/// `runs` is a list of (start_index, length); a run is clipped to the
/// buffer, and one starting at or past its end does nothing.
pub fn add_burst_dropouts(
    iq: &mut [Complex<f32>],
    runs: &[(usize, usize)],
    sigma: f32,
    seed: &mut u64,
) {
    for &(start, len) in runs {
        let end = start.saturating_add(len).min(iq.len());
        if start >= end {
            continue;
        }
        for z in &mut iq[start..end] {
            *z = Complex::new(
                sigma * crate::synthetic::gaussian_noise(seed),
                sigma * crate::synthetic::gaussian_noise(seed),
            );
        }
    }
}

/// Applies a slow fade envelope (e.g. sinusoidal variation in amplitude).
/// `frequency_hz` is the fade frequency, `depth` is the fade depth (0.0 to 1.0).
pub fn add_slow_fade(iq: &mut [Complex<f32>], sample_rate: f32, frequency_hz: f32, depth: f32) {
    let omega = 2.0 * std::f32::consts::PI * frequency_hz / sample_rate;
    for (i, z) in iq.iter_mut().enumerate() {
        let envelope = 1.0 - depth * (1.0 - (omega * i as f32).cos()) * 0.5;
        *z *= envelope;
    }
}

/// Randomly injects high-amplitude spikes to simulate motor/ESC spark noise.
/// `probability` (0.0 to 1.0) is the chance per sample of a spike occurring.
/// `amplitude` is the maximum magnitude of the spike.
pub fn add_impulsive_noise(
    iq: &mut [Complex<f32>],
    probability: f32,
    amplitude: f32,
    seed: &mut u64,
) {
    for z in iq.iter_mut() {
        // A uniform draw, so `P(spike) = probability`. The mean of twelve
        // uniforms (what `gaussian_noise` sums) is concentrated around
        // 0.5 and almost never falls below a small probability.
        let rand_val = crate::synthetic::xorshift01(seed);
        if rand_val < probability {
            let phase = crate::synthetic::gaussian_noise(seed) * std::f32::consts::PI;
            z.re += amplitude * phase.cos();
            z.im += amplitude * phase.sin();
        }
    }
}

/// Erases demod-domain sync pulses to blanking level across the specified sample runs `(start, len)`.
/// Runs are clipped to the buffer; one starting at or past its end does nothing.
pub fn apply_demod_blanking(demod: &mut [f32], runs: &[(usize, usize)], blank_level: f32) {
    for &(start, len) in runs {
        let end = start.saturating_add(len).min(demod.len());
        if start < end {
            demod[start..end].fill(blank_level);
        }
    }
}

/// Applies attenuation fades to demod-domain samples over `(start, len, attenuation_factor)` runs.
/// Runs are clipped to the buffer; one starting at or past its end does nothing.
pub fn apply_demod_fade(demod: &mut [f32], runs: &[(usize, usize, f32)]) {
    for &(start, len, atten) in runs {
        let end = start.saturating_add(len).min(demod.len());
        if start >= end {
            continue;
        }
        for sample in &mut demod[start..end] {
            *sample *= atten;
        }
    }
}

/// Injects additive Gaussian noise into demod-domain samples over the specified `(start, len, sigma)` intervals.
/// Runs are clipped to the buffer; one starting at or past its end does nothing.
pub fn apply_demod_awgn(demod: &mut [f32], runs: &[(usize, usize, f32)], seed: &mut u64) {
    for &(start, len, sigma) in runs {
        if sigma <= 0.0 {
            continue;
        }
        let end = start.saturating_add(len).min(demod.len());
        if start >= end {
            continue;
        }
        for sample in &mut demod[start..end] {
            *sample += sigma * crate::synthetic::gaussian_noise(seed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `probability` is the per-sample spike rate. Compared against a
    /// mean of twelve uniforms instead of one uniform draw, a 5% setting
    /// would fire ~4.5e-12 of the time and the impairment would be plain
    /// AWGN.
    #[test]
    fn impulsive_noise_fires_at_the_requested_rate() {
        let n = 200_000;
        let mut iq = vec![Complex::new(0.0f32, 0.0); n];
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        add_impulsive_noise(&mut iq, 0.05, 10.0, &mut seed);
        let spikes = iq.iter().filter(|z| z.norm() > 1.0).count();
        let rate = spikes as f64 / n as f64;
        assert!((rate - 0.05).abs() < 0.005, "spike rate {rate}");
    }

    /// A run starting past the end of the buffer is a no-op, as it is for
    /// `apply_demod_blanking`, rather than a slice panic.
    #[test]
    fn runs_past_the_end_are_ignored() {
        let mut seed = 7u64;
        let mut iq = vec![Complex::new(1.0f32, 0.0); 8];
        add_burst_dropouts(&mut iq, &[(20, 4), (8, 1), (usize::MAX, 2)], 1.0, &mut seed);
        assert!(iq.iter().all(|z| *z == Complex::new(1.0, 0.0)));

        let mut demod = vec![1.0f32; 8];
        apply_demod_fade(&mut demod, &[(20, 4, 0.5), (usize::MAX, 2, 0.5)]);
        apply_demod_awgn(&mut demod, &[(20, 4, 1.0), (usize::MAX, 2, 1.0)], &mut seed);
        apply_demod_blanking(&mut demod, &[(20, 4), (usize::MAX, 2)], 0.0);
        assert!(demod.iter().all(|&v| v == 1.0));

        // A run straddling the end is clipped, not skipped.
        apply_demod_fade(&mut demod, &[(6, 10, 0.5)]);
        assert_eq!(demod, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]);
    }
}
