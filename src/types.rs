use crate::bands::FpvChannel;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Largest I or Q magnitude the crate's entry points let through.
///
/// A finite sample can still be poison: the discriminator multiplies
/// neighbouring samples, so anything past about 1.8e19 overflows `f32`
/// to `inf - inf` and comes out NaN, which then sticks in every filter
/// with memory downstream. 1e9 is far above any real capture (a
/// normalised stream peaks near 1, an unscaled 16-bit one at 32,768)
/// and far below where squares and FIR sums overflow.
pub const MAX_IQ_COMPONENT: f32 = 1.0e9;

/// One sample made safe to process: zero if either part is NaN or
/// infinite, otherwise each part clamped to ±[`MAX_IQ_COMPONENT`].
#[inline]
pub fn sanitize_iq_sample(s: Complex<f32>) -> Complex<f32> {
    if s.re.is_finite() && s.im.is_finite() {
        Complex::new(
            s.re.clamp(-MAX_IQ_COMPONENT, MAX_IQ_COMPONENT),
            s.im.clamp(-MAX_IQ_COMPONENT, MAX_IQ_COMPONENT),
        )
    } else {
        Complex::new(0.0, 0.0)
    }
}

/// `iq` with every sample passed through [`sanitize_iq_sample`].
///
/// Borrows when nothing needs changing, which is every real capture, so
/// the cost on the hot path is one comparison pass and no copy.
pub fn sanitize_iq(iq: &[Complex<f32>]) -> Cow<'_, [Complex<f32>]> {
    // One test covers NaN, infinity and out-of-range: every comparison
    // with NaN is false.
    let sane = |s: &Complex<f32>| s.re.abs() <= MAX_IQ_COMPONENT && s.im.abs() <= MAX_IQ_COMPONENT;
    if iq.iter().all(sane) {
        Cow::Borrowed(iq)
    } else {
        Cow::Owned(iq.iter().copied().map(sanitize_iq_sample).collect())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub channel: Option<FpvChannel>,
    pub frequency_hz: u64,
    pub confidence: f32, // 0.0 to 1.0
    /// Mean signal power as `10·log10(mean |IQ|²)` — dB relative to
    /// digital full scale (dBFS), *not* calibrated dBm: no SDR gain or
    /// antenna factor is applied. Comparable between detections from
    /// the same capture chain, not across devices. (Field name kept
    /// for serialization compatibility.)
    pub rssi_dbm: f32,
    pub bandwidth_hz: u32,
    pub signal_type: SignalType,
}

/// The enum is `#[non_exhaustive]` so future variants (e.g. digital
/// FPV protocols, RC telemetry tags) can be added without breaking
/// downstream matches. Callers must include a wildcard arm or
/// exhaustively match the variants they care about.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignalType {
    /// No identifiable signal — silence, noise, or an unrecognised
    /// pattern. `detect_from_iq` filters these out before returning.
    Unknown,
    /// Confirmed analog FM video with NTSC line/field rates
    /// (15734 Hz / 60 Hz).
    AnalogVideoNtsc,
    /// Confirmed analog FM video with PAL line/field rates
    /// (15625 Hz / 50 Hz).
    AnalogVideoPal,
    /// Analog FM video detected (strong H-sync line-rate energy in
    /// the FM-demodulated spectrum) but the FFT bin resolution was
    /// too coarse to discriminate the 15625 Hz / 15734 Hz line
    /// spacing — which happens whenever per-bin resolution is
    /// ≥ ~109 Hz. The caller knows analog FPV is present but should
    /// not act on a PAL-vs-NTSC tag. Earlier code silently defaulted
    /// to NTSC here, which produced a false standards tag whenever
    /// the FFT couldn't separate the two line rates.
    AnalogVideoUnknown,
    /// Wideband digital signal, likely Wi-Fi / Bluetooth / LTE.
    WidebandDigital,
    /// Narrowband interferer (CW tone, single-carrier emitter).
    NarrowbandInterference,
}

impl SignalType {
    /// `true` for any of `AnalogVideoNtsc` / `AnalogVideoPal` /
    /// `AnalogVideoUnknown`. Callers should use this when they want
    /// to gate on "is this an analog FPV detection?" rather than
    /// `!= Unknown`, which would also accept `WidebandDigital` and
    /// `NarrowbandInterference`.
    pub fn is_analog_video(self) -> bool {
        matches!(
            self,
            SignalType::AnalogVideoNtsc
                | SignalType::AnalogVideoPal
                | SignalType::AnalogVideoUnknown,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_block_is_borrowed_untouched() {
        let iq = vec![
            Complex::new(0.5, -0.5),
            Complex::new(-32_768.0, 32_767.0),
            Complex::new(MAX_IQ_COMPONENT, -MAX_IQ_COMPONENT),
            Complex::new(0.0, -0.0),
        ];
        let out = sanitize_iq(&iq);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(&*out, &iq[..]);
        assert!(matches!(sanitize_iq(&[]), Cow::Borrowed(_)));
    }

    #[test]
    fn hostile_samples_are_zeroed_or_clamped() {
        let iq = [
            Complex::new(1.0, 2.0),
            Complex::new(f32::NAN, 2.0),
            Complex::new(1.0, f32::INFINITY),
            Complex::new(f32::NEG_INFINITY, f32::NAN),
            Complex::new(3.0e38, -1.0e30),
            Complex::new(-2.0e9, 7.0),
        ];
        let out = sanitize_iq(&iq);
        assert_eq!(
            &*out,
            &[
                Complex::new(1.0, 2.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(MAX_IQ_COMPONENT, -MAX_IQ_COMPONENT),
                Complex::new(-MAX_IQ_COMPONENT, 7.0),
            ]
        );
        // What the bound is for: neighbouring samples multiply in the
        // discriminator, and that product must stay finite.
        let worst = Complex::new(MAX_IQ_COMPONENT, MAX_IQ_COMPONENT);
        assert!((worst * worst.conj()).arg().is_finite());
        assert!(
            (worst * Complex::new(-MAX_IQ_COMPONENT, MAX_IQ_COMPONENT))
                .arg()
                .is_finite()
        );
    }
}
