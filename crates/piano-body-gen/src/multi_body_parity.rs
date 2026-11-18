//! Sign-invariant, cross-CPU / cross-`libm` parity verification for `.pbdy` presets.

use crate::bake::BakeDiagnostics;
use crate::pole_cancel::verify_pole_cancellation;
use crate::presets::BodyArchetype;
use piano_bridge::{KEYBED_ZONE_IDX, NUM_BRIDGE_ZONES};
use piano_spec::{BodyPreset, PRESET_RAD_COEFFS};

fn assert_approx(label: &str, a: f32, b: f32, rel_tol: f32, abs_tol: f32) {
    assert!(
        a.is_finite()
            && b.is_finite()
            && (a - b).abs() <= abs_tol.max(rel_tol * a.abs().max(b.abs())),
        "{label} mismatch: disk={a} vs baked={b}"
    );
}

fn assert_slice_approx(label: &str, a: &[f32], b: &[f32], rel_tol: f32, abs_tol: f32) {
    assert_eq!(a.len(), b.len(), "{label} length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_approx(&format!("{label}[{i}]"), x, y, rel_tol, abs_tol);
    }
}

fn max_abs(slice: &[f32]) -> f32 {
    slice.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()))
}

pub(crate) fn assert_sign_invariant_pbdy_parity(
    arch: BodyArchetype,
    decoded: &BodyPreset,
    baked: &BodyPreset,
    diag: &BakeDiagnostics,
) {
    let slug = arch.slug();
    let baked_rt = BodyPreset::decode(&baked.encode())
        .unwrap_or_else(|e| panic!("{slug} baked roundtrip decode failed: {e}"));

    assert_eq!(decoded.modes.len(), diag.total_modes);
    assert_eq!(
        decoded.modes.len(),
        baked_rt.modes.len(),
        "{slug} modes len"
    );

    assert_slice_approx(
        &format!("{slug} y_inf"),
        &decoded.y_inf,
        &baked_rt.y_inf,
        1.0e-5,
        1.0e-9,
    );
    assert_slice_approx(
        &format!("{slug} direct_gain"),
        &decoded.direct_zone_gain,
        &baked_rt.direct_zone_gain,
        1.0e-5,
        1.0e-7,
    );
    for (z, (dp, bp)) in decoded
        .zone_pos_m
        .iter()
        .zip(baked_rt.zone_pos_m.iter())
        .enumerate()
    {
        assert_slice_approx(&format!("{slug} zone_pos[{z}]"), dp, bp, 1.0e-5, 1.0e-6);
    }
    for (p, (dp, bp)) in decoded
        .reduced_poles
        .iter()
        .zip(baked_rt.reduced_poles.iter())
        .enumerate()
    {
        assert_approx(
            &format!("{slug} pole[{p}].f"),
            dp.freq_hz,
            bp.freq_hz,
            1.0e-5,
            1.0e-5,
        );
        assert_approx(
            &format!("{slug} pole[{p}].z"),
            dp.zeta,
            bp.zeta,
            1.0e-5,
            1.0e-7,
        );
        assert_approx(
            &format!("{slug} pole[{p}].m"),
            dp.modal_mass_kg,
            bp.modal_mass_kg,
            1.0e-5,
            1.0e-6,
        );
        assert_slice_approx(
            &format!("{slug} pole[{p}].g"),
            &dp.zone_gains,
            &bp.zone_gains,
            1.0e-5,
            1.0e-7,
        );
    }

    // Across x86_64 CPUs and glibc libm IFUNCs (e.g. FMA vs non-FMA), 1-ULP trig/hypot
    // differences can flip arbitrary +/-1 eigenvector signs in the 300x300 Ritz solve.
    // Every mode must still match in freq, zeta, cavity_flag, quantization scale,
    // eigenvector-independent keybed drive, and coupled +/-1 sign-aligned (zone_drive[0..8], rad_coeffs).
    for (m, (d, b)) in decoded.modes.iter().zip(baked_rt.modes.iter()).enumerate() {
        assert_eq!(d.cavity_flag, b.cavity_flag, "{slug} mode {m} cavity_flag");
        assert_approx(
            &format!("{slug} mode[{m}].f"),
            d.freq_hz,
            b.freq_hz,
            1.0e-5,
            1.0e-5,
        );
        assert_approx(
            &format!("{slug} mode[{m}].z"),
            d.zeta,
            b.zeta,
            1.0e-5,
            1.0e-8,
        );

        let (max_d_disk, max_d_baked) = (max_abs(&d.zone_drive), max_abs(&b.zone_drive));
        let (max_r_disk, max_r_baked) = (max_abs(&d.rad_coeffs), max_abs(&b.rad_coeffs));
        assert_approx(
            &format!("{slug} mode[{m}].scale_d"),
            max_d_disk,
            max_d_baked,
            1.0e-5,
            1.0e-8,
        );
        assert_approx(
            &format!("{slug} mode[{m}].scale_r"),
            max_r_disk,
            max_r_baked,
            1.0e-5,
            1.0e-8,
        );

        let dot: f64 = d.zone_drive[..NUM_BRIDGE_ZONES]
            .iter()
            .zip(b.zone_drive[..NUM_BRIDGE_ZONES].iter())
            .chain(d.rad_coeffs.iter().zip(b.rad_coeffs.iter()))
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let sign = if dot >= 0.0 { 1.0_f32 } else { -1.0_f32 };

        let tol_d = (5.0e-5 * max_d_disk.max(max_d_baked)).max(1.0e-7);
        for z in 0..NUM_BRIDGE_ZONES {
            let err = (d.zone_drive[z] - sign * b.zone_drive[z]).abs();
            assert!(
                err <= tol_d,
                "{slug} mode[{m}].zd[{z}] (sign={sign}): err={err} > {tol_d}"
            );
        }
        // Keybed drive (Zone 8) is independent of eigenvectors on <360 Hz Ritz modes and 0 above.
        let kb_err = (d.zone_drive[KEYBED_ZONE_IDX] - b.zone_drive[KEYBED_ZONE_IDX]).abs();
        assert!(
            kb_err <= tol_d,
            "{slug} mode[{m}].zd[KEYBED]: err={kb_err} > {tol_d}"
        );

        let tol_r = (5.0e-5 * max_r_disk.max(max_r_baked)).max(1.0e-7);
        for r in 0..PRESET_RAD_COEFFS {
            let err = (d.rad_coeffs[r] - sign * b.rad_coeffs[r]).abs();
            assert!(
                err <= tol_r,
                "{slug} mode[{m}].rc[{r}] (sign={sign}): err={err} > {tol_r}"
            );
        }
    }

    let rep_disk = verify_pole_cancellation(decoded, 48_000.0);
    let rep_baked = verify_pole_cancellation(baked, 48_000.0);
    assert!(
        rep_disk.max_inverse_pole_real < -1.0
            && rep_disk.max_matrix_residual < 1.0e-6
            && rep_disk.max_cancelled_excess_db <= 1.0
            && rep_disk.max_discrete_cascade_err_db <= 1.0,
        "{slug} decoded pole cancellation failed: {rep_disk:?}"
    );
    assert!(
        (rep_disk.max_cancelled_excess_db - rep_baked.max_cancelled_excess_db).abs() <= 1.0e-4,
        "{slug} cancel_err drift: disk={} vs baked={}",
        rep_disk.max_cancelled_excess_db,
        rep_baked.max_cancelled_excess_db
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::cached_concert_grand;

    fn baked_roundtrip_clone() -> (BodyPreset, &'static BodyPreset, &'static BakeDiagnostics) {
        let (baked, diag) = cached_concert_grand();
        let decoded = BodyPreset::decode(&baked.encode()).expect("decode baked");
        (decoded, baked, diag)
    }

    #[test]
    fn parity_accepts_coupled_eigenvector_sign_flip_on_bridge_and_rad() {
        let (mut decoded, baked, diag) = baked_roundtrip_clone();
        let m = decoded
            .modes
            .iter()
            .position(|m| m.freq_hz < 360.0 && m.zone_drive[KEYBED_ZONE_IDX].abs() > 1.0e-6)
            .expect("low-frequency mode with keybed drive");
        for z in 0..NUM_BRIDGE_ZONES {
            decoded.modes[m].zone_drive[z] = -decoded.modes[m].zone_drive[z];
        }
        for r in 0..PRESET_RAD_COEFFS {
            decoded.modes[m].rad_coeffs[r] = -decoded.modes[m].rad_coeffs[r];
        }
        assert_sign_invariant_pbdy_parity(BodyArchetype::ConcertGrand, &decoded, baked, diag);
    }

    #[test]
    #[should_panic(expected = "mode[")]
    fn parity_rejects_decoupled_bridge_flip_without_rad_flip() {
        let (mut decoded, baked, diag) = baked_roundtrip_clone();
        let m = decoded
            .modes
            .iter()
            .position(|m| max_abs(&m.rad_coeffs) > 0.01 && max_abs(&m.zone_drive) > 0.001)
            .expect("active mode");
        for z in 0..NUM_BRIDGE_ZONES {
            decoded.modes[m].zone_drive[z] = -decoded.modes[m].zone_drive[z];
        }
        assert_sign_invariant_pbdy_parity(BodyArchetype::ConcertGrand, &decoded, baked, diag);
    }

    #[test]
    #[should_panic(expected = "zd[KEYBED]")]
    fn parity_rejects_keybed_sign_flip() {
        let (mut decoded, baked, diag) = baked_roundtrip_clone();
        let m = decoded
            .modes
            .iter()
            .position(|m| m.freq_hz < 360.0 && m.zone_drive[KEYBED_ZONE_IDX].abs() > 1.0e-6)
            .expect("low-frequency mode with keybed drive");
        decoded.modes[m].zone_drive[KEYBED_ZONE_IDX] =
            -decoded.modes[m].zone_drive[KEYBED_ZONE_IDX];
        assert_sign_invariant_pbdy_parity(BodyArchetype::ConcertGrand, &decoded, baked, diag);
    }
}
