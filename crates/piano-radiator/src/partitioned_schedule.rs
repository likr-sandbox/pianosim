//! Staggered `4x`-growth NUPOLS tier slot scheduling (`P0-3` collision-free Tier 3).

use piano_simd::AlignedBuf;
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::nupols_tier::NupolsTier;

pub(crate) struct TierFireCtx<'a> {
    pub(crate) in_ring: &'a [AlignedBuf; PRESET_RADIATOR_ZONES],
    pub(crate) out_ring_l: &'a mut AlignedBuf,
    pub(crate) out_ring_r: &'a mut AlignedBuf,
    pub(crate) sample_pos: usize,
    pub(crate) m: usize,
    pub(crate) delay: usize,
    pub(crate) work: &'a mut [f32],
}

pub(crate) fn fire_tier_256(tier: &mut NupolsTier, ctx: &mut TierFireCtx<'_>) {
    let sub = ctx.m & 3;
    if sub == 0 && ctx.m >= 4 {
        tier.step_fdl_head();
        tier.pending_s_in_end = ctx.sample_pos;
        tier.has_pending = true;
        tier.forward_zones_range(ctx.in_ring, ctx.sample_pos, 0, 4, ctx.work);
    } else if tier.has_pending {
        let s_in_end = tier.pending_s_in_end;
        match sub {
            1 => tier.forward_zones_range(ctx.in_ring, s_in_end, 4, 9, ctx.work),
            2 => {
                tier.clear_acc();
                tier.accumulate_partitions(0, tier.p_count);
            }
            3 => {
                tier.inverse_channel_add(0, ctx.out_ring_l, s_in_end, ctx.delay, ctx.work);
                tier.inverse_channel_add(1, ctx.out_ring_r, s_in_end, ctx.delay, ctx.work);
                tier.has_pending = false;
            }
            _ => {}
        }
    }
}

pub(crate) fn fire_tier_1024(tier: &mut NupolsTier, ctx: &mut TierFireCtx<'_>) {
    let sub = ctx.m & 15;
    if sub == 0 && ctx.m >= 16 {
        tier.step_fdl_head();
        tier.pending_s_in_end = ctx.sample_pos;
        tier.has_pending = true;
        tier.forward_zones_range(ctx.in_ring, ctx.sample_pos, 0, 1, ctx.work);
    } else if tier.has_pending {
        let s_in_end = tier.pending_s_in_end;
        let p1 = tier.p_count / 3;
        let p2 = (2 * tier.p_count) / 3;
        match sub {
            1..=7 => tier.forward_zones_range(ctx.in_ring, s_in_end, sub, sub + 1, ctx.work),
            9 => tier.forward_zones_range(ctx.in_ring, s_in_end, 8, 9, ctx.work),
            10 => {
                tier.clear_acc();
                tier.accumulate_partitions(0, p1);
            }
            11 => tier.accumulate_partitions(p1, p2),
            12 => tier.accumulate_partitions(p2, tier.p_count),
            13 => tier.inverse_channel_add(0, ctx.out_ring_l, s_in_end, ctx.delay, ctx.work),
            15 => {
                tier.inverse_channel_add(1, ctx.out_ring_r, s_in_end, ctx.delay, ctx.work);
                tier.has_pending = false;
            }
            _ => {}
        }
    }
}

/// `P0-3` collision-free even-slot schedule: never collides with Tier 1 (`m&3 in {1,3}`)
/// or Tier 2 accum/IFFT (`m&15 in {10,11,12,13,15}`), placing both IFFTs on Tier 2 idle slots.
pub(crate) fn fire_tier_4096(tier: &mut NupolsTier, ctx: &mut TierFireCtx<'_>) {
    let sub = ctx.m & 63;
    if sub == 0 && ctx.m >= 64 {
        tier.step_fdl_head();
        tier.pending_s_in_end = ctx.sample_pos;
        tier.has_pending = true;
    } else if tier.has_pending {
        let s_in_end = tier.pending_s_in_end;
        let p1 = tier.p_count / 4;
        let p2 = (2 * tier.p_count) / 4;
        let p3 = (3 * tier.p_count) / 4;
        match sub {
            2 => tier.forward_zones_range(ctx.in_ring, s_in_end, 0, 1, ctx.work),
            6 => tier.forward_zones_range(ctx.in_ring, s_in_end, 1, 2, ctx.work),
            8 => tier.forward_zones_range(ctx.in_ring, s_in_end, 2, 3, ctx.work),
            14 => tier.forward_zones_range(ctx.in_ring, s_in_end, 3, 4, ctx.work),
            18 => tier.forward_zones_range(ctx.in_ring, s_in_end, 4, 5, ctx.work),
            22 => tier.forward_zones_range(ctx.in_ring, s_in_end, 5, 6, ctx.work),
            24 => tier.forward_zones_range(ctx.in_ring, s_in_end, 6, 7, ctx.work),
            30 => tier.forward_zones_range(ctx.in_ring, s_in_end, 7, 8, ctx.work),
            38 => tier.forward_zones_range(ctx.in_ring, s_in_end, 8, 9, ctx.work),
            40 => {
                tier.clear_acc();
                tier.accumulate_partitions(0, p1);
            }
            46 => tier.accumulate_partitions(p1, p2),
            50 => tier.accumulate_partitions(p2, p3),
            54 => tier.accumulate_partitions(p3, tier.p_count),
            56 => tier.inverse_channel_add(0, ctx.out_ring_l, s_in_end, ctx.delay, ctx.work),
            62 => {
                tier.inverse_channel_add(1, ctx.out_ring_r, s_in_end, ctx.delay, ctx.work);
                tier.has_pending = false;
            }
            _ => {}
        }
    }
}
