//! Fidelity metrics for reconstructed video frames.

/// Computes Peak Signal-to-Noise Ratio (PSNR) between a test frame and a ground truth frame.
/// Both frames should be packed `u32` (e.g. ARGB/XRGB). We extract the lower 8 bits (Luma) for comparison.
///
/// Returns `f64::INFINITY` for a perfect match, and for two empty frames,
/// which have no pixel that differs.
pub fn compute_psnr(test: &[u32], truth: &[u32]) -> f64 {
    assert_eq!(test.len(), truth.len());
    if test.is_empty() {
        return f64::INFINITY;
    }
    let mut mse = 0.0f64;
    for (a, b) in test.iter().zip(truth.iter()) {
        let diff = (a & 0xFF) as f64 - (b & 0xFF) as f64;
        mse += diff * diff;
    }
    mse /= test.len() as f64;
    if mse < 1e-10 {
        return f64::INFINITY; // Perfect match
    }
    10.0 * (255.0f64 * 255.0f64 / mse).log10()
}

/// Computes the Pearson correlation coefficient of spatial gradients (Sobel-like) between two frames.
/// The frames must have the given `width` and `height`.
///
/// Returns 0.0 when there are no gradients to correlate (either dimension
/// below 2, including an empty frame) or when either frame's gradients
/// have zero variance.
pub fn compute_gradient_correlation(
    test: &[u32],
    truth: &[u32],
    width: usize,
    height: usize,
) -> f64 {
    assert_eq!(test.len(), width * height);
    assert_eq!(truth.len(), width * height);

    // We compute simple forward differences (dx, dy) to get the gradient magnitude.
    let mut mean_test = 0.0f64;
    let mut mean_truth = 0.0f64;

    // Forward differences need a right and a lower neighbour, so the
    // gradient grid is one smaller than the frame each way; saturating
    // keeps a zero dimension at an empty grid instead of underflowing.
    let grad_w = width.saturating_sub(1);
    let grad_h = height.saturating_sub(1);
    let mut grads_test = Vec::with_capacity(grad_w * grad_h);
    let mut grads_truth = Vec::with_capacity(grad_w * grad_h);

    for y in 0..grad_h {
        for x in 0..grad_w {
            let idx = y * width + x;

            let t_xy = (test[idx] & 0xFF) as f64;
            let t_x1y = (test[idx + 1] & 0xFF) as f64;
            let t_xy1 = (test[idx + width] & 0xFF) as f64;
            let grad_test = ((t_x1y - t_xy).powi(2) + (t_xy1 - t_xy).powi(2)).sqrt();
            grads_test.push(grad_test);
            mean_test += grad_test;

            let r_xy = (truth[idx] & 0xFF) as f64;
            let r_x1y = (truth[idx + 1] & 0xFF) as f64;
            let r_xy1 = (truth[idx + width] & 0xFF) as f64;
            let grad_truth = ((r_x1y - r_xy).powi(2) + (r_xy1 - r_xy).powi(2)).sqrt();
            grads_truth.push(grad_truth);
            mean_truth += grad_truth;
        }
    }

    let n = grads_test.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    mean_test /= n;
    mean_truth /= n;

    let mut cov = 0.0f64;
    let mut var_test = 0.0f64;
    let mut var_truth = 0.0f64;

    for (gt, gr) in grads_test.iter().zip(grads_truth.iter()) {
        let dt = gt - mean_test;
        let dr = gr - mean_truth;
        cov += dt * dr;
        var_test += dt * dt;
        var_truth += dr * dr;
    }

    if var_test == 0.0 || var_truth == 0.0 {
        return 0.0;
    }

    cov / (var_test * var_truth).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_frames_return_documented_values() {
        assert_eq!(compute_psnr(&[], &[]), f64::INFINITY);
        assert_eq!(compute_gradient_correlation(&[], &[], 0, 0), 0.0);
        assert_eq!(compute_gradient_correlation(&[], &[], 0, 7), 0.0);
        assert_eq!(compute_gradient_correlation(&[], &[], 7, 0), 0.0);
        assert_eq!(
            compute_gradient_correlation(&[1, 2, 3], &[1, 2, 3], 3, 1),
            0.0
        );
    }

    #[test]
    fn identical_textured_frames_correlate_fully() {
        let frame: Vec<u32> = (0..16u32).map(|i| (i * 37) % 251).collect();
        assert_eq!(compute_psnr(&frame, &frame), f64::INFINITY);
        let r = compute_gradient_correlation(&frame, &frame, 4, 4);
        assert!((r - 1.0).abs() < 1e-12, "r = {r}");
    }
}
