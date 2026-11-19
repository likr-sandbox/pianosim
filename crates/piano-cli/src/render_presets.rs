//! Showcase demo manifest, WAV format, and preset configurations for `render-wav`/`render-demos`.

use piano_engine::{
    BodyArchetype, PerfProfile, PianoEngine, DEFAULT_REVERB_DECAY_S, DEFAULT_REVERB_MIX,
};
use piano_spec::voicing::A4_FREQ_DEFAULT;
use piano_spec::{LidState, ListenerPerspective, Temperament, VoicingSpec};

pub(crate) struct ShowcaseDemo {
    pub(crate) title: &'static str,
    pub(crate) mid_filename: &'static str,
    pub(crate) wav_filename: &'static str,
}

pub(crate) const SHOWCASE_DEMOS: [ShowcaseDemo; 6] = [
    ShowcaseDemo {
        title: "Bach: Prelude in C (BWV 846)",
        mid_filename: "01-bach-prelude-bwv846.mid",
        wav_filename: "01-bach-prelude-bwv846.wav",
    },
    ShowcaseDemo {
        title: "Chopin: Nocturne in Eb (Op. 9 No. 2)",
        mid_filename: "02-chopin-nocturne-op9-no2.mid",
        wav_filename: "02-chopin-nocturne-op9-no2.wav",
    },
    ShowcaseDemo {
        title: "Beethoven: Moonlight Sonata (Op. 27 No. 2)",
        mid_filename: "03-beethoven-moonlight-op27-no2.mid",
        wav_filename: "03-beethoven-moonlight-op27-no2.wav",
    },
    ShowcaseDemo {
        title: "Debussy: Clair de Lune",
        mid_filename: "04-debussy-clair-de-lune.mid",
        wav_filename: "04-debussy-clair-de-lune.wav",
    },
    ShowcaseDemo {
        title: "Satie: Gymnopédie No. 1",
        mid_filename: "05-satie-gymnopedie-no1.mid",
        wav_filename: "05-satie-gymnopedie-no1.wav",
    },
    ShowcaseDemo {
        title: "Rachmaninoff: Prelude in C# Minor (Op. 3 No. 2)",
        mid_filename: "06-rachmaninoff-prelude-op3-no2.mid",
        wav_filename: "06-rachmaninoff-prelude-op3-no2.wav",
    },
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum WavFormat {
    Pcm16,
    Pcm24,
    F32,
}

impl WavFormat {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Pcm16 => "pcm16",
            Self::Pcm24 => "pcm24",
            Self::F32 => "f32",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PresetId {
    ConcertGrand,
    IntimateUpright,
    BrightPop,
    VintageSaloon,
}

impl PresetId {
    pub(crate) fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "concert-grand" => Ok(Self::ConcertGrand),
            "intimate-upright" => Ok(Self::IntimateUpright),
            "bright-pop" => Ok(Self::BrightPop),
            "vintage-saloon" => Ok(Self::VintageSaloon),
            _ => Err(format!(
                "invalid --preset '{raw}' (expected concert-grand, intimate-upright, bright-pop, or vintage-saloon)"
            )),
        }
    }

    pub(crate) const fn slug(self) -> &'static str {
        match self {
            Self::ConcertGrand => "concert-grand",
            Self::IntimateUpright => "intimate-upright",
            Self::BrightPop => "bright-pop",
            Self::VintageSaloon => "vintage-saloon",
        }
    }

    pub(crate) fn configure_engine(
        self,
        sample_rate: u32,
        a4_override: Option<f32>,
        profile: PerfProfile,
    ) -> (PianoEngine, f32) {
        let (hh, ud, st, sl, a4, cond, body, persp, lid, decay_s, mix) = match self {
            Self::ConcertGrand => (
                0.50,
                1.00,
                1.00,
                1.00,
                A4_FREQ_DEFAULT,
                0.0,
                BodyArchetype::ConcertGrand,
                ListenerPerspective::Player,
                LidState::Open,
                DEFAULT_REVERB_DECAY_S,
                DEFAULT_REVERB_MIX,
            ),
            Self::IntimateUpright => (
                0.25,
                0.60,
                0.70,
                0.75,
                A4_FREQ_DEFAULT,
                0.0,
                BodyArchetype::UprightStudio,
                ListenerPerspective::CloseStereo,
                LidState::Closed,
                0.85,
                0.12,
            ),
            Self::BrightPop => (
                0.80,
                1.40,
                1.30,
                1.15,
                442.0,
                0.0,
                BodyArchetype::ConcertGrand,
                ListenerPerspective::CloseStereo,
                LidState::Open,
                1.25,
                0.22,
            ),
            Self::VintageSaloon => (
                0.70,
                2.50,
                0.20,
                0.65,
                435.0,
                0.25,
                BodyArchetype::VintageSaloon,
                ListenerPerspective::Player,
                LidState::Half,
                0.55,
                0.08,
            ),
        };
        let mut voicing =
            VoicingSpec::from_globals_full(hh, ud, st, sl, a4, Temperament::Equal, cond);
        if let Some(a4_hz) = a4_override {
            voicing = voicing.with_concert_pitch_hz(a4_hz);
        }
        let eff_a4 = voicing.effective_a4_freq_hz();
        let mut engine = PianoEngine::new_with_profile(sample_rate as f32, voicing, profile);
        engine.set_body_space(body, persp, lid);
        engine.set_reverb(decay_s, mix);
        (engine, eff_a4)
    }
}
