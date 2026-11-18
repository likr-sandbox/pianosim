/** Physical modeling parameter presets for PianoSim.
 * Encapsulates macroscopic Voicing and Acoustic Space configurations into distinct piano archetypes.
 */

import {
  BODY_CONCERT_GRAND,
  BODY_UPRIGHT_STUDIO,
  BODY_VINTAGE_SALOON,
  LID_CLOSED,
  LID_HALF,
  LID_OPEN,
  PERSPECTIVE_CLOSE,
  PERSPECTIVE_PLAYER,
  REVERB_DECAY_DEFAULT,
  REVERB_MIX_DEFAULT,
  type BodyId,
  type LidState,
  type ListenerPerspective,
} from '../audio/abi';

export interface ParameterPreset {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly voicing: {
    readonly hammerHardness: number;
    readonly unisonDetune: number;
    readonly stretchTuning: number;
    readonly stringLengthScale: number;
    readonly a4FreqHz: number;
  };
  readonly space: {
    readonly bodyId: BodyId;
    readonly perspective: ListenerPerspective; // ListenerPerspective (0: Player, 1: Close, 2: Hall, 3: Binaural)
    readonly lidState: LidState;                // LidState (0: Closed, 1: Half, 2: Open)
    readonly reverbDecayS: number;
    readonly reverbMix: number;
  };
}

export const PIANO_PRESETS: ReadonlyArray<ParameterPreset> = [
  {
    id: 'concert-grand',
    name: 'Concert Grand',
    description: '9-foot D-274 reference setup with balanced Railsback stretch, natural unison bloom, and open lid projection.',
    voicing: {
      hammerHardness: 0.50,
      unisonDetune: 1.00,
      stretchTuning: 1.00,
      stringLengthScale: 1.00,
      a4FreqHz: 440.0,
    },
    space: {
      bodyId: BODY_CONCERT_GRAND,
      perspective: PERSPECTIVE_PLAYER, // 0
      lidState: LID_OPEN,              // 2
      reverbDecayS: REVERB_DECAY_DEFAULT, // 1.6 s
      reverbMix: REVERB_MIX_DEFAULT,      // 0.18
    },
  },
  {
    id: 'intimate-upright',
    name: 'Intimate Upright',
    description: 'Warm & mellow chamber character with soft voiced felt, shortened string scale, and closed lid damping.',
    voicing: {
      hammerHardness: 0.25,
      unisonDetune: 0.60,
      stretchTuning: 0.70,
      stringLengthScale: 0.75,
      a4FreqHz: 440.0,
    },
    space: {
      bodyId: BODY_UPRIGHT_STUDIO,
      perspective: PERSPECTIVE_CLOSE,  // 1
      lidState: LID_CLOSED,            // 0
      reverbDecayS: 0.85,
      reverbMix: 0.12,
    },
  },
  {
    id: 'bright-pop',
    name: 'Bright Pop',
    description: 'Cutting modern studio sound with hardened lacquer hammers, 442 Hz brilliance, and wide unison chorusing.',
    voicing: {
      hammerHardness: 0.80,
      unisonDetune: 1.40,
      stretchTuning: 1.30,
      stringLengthScale: 1.15,
      a4FreqHz: 442.0,
    },
    space: {
      bodyId: BODY_CONCERT_GRAND,
      perspective: PERSPECTIVE_CLOSE,  // 1
      lidState: LID_OPEN,              // 2
      reverbDecayS: 1.25,
      reverbMix: 0.22,
    },
  },
  {
    id: 'vintage-saloon',
    name: 'Vintage Saloon',
    description: 'Historic 435 Hz fortepiano & saloon character with dry snappy impact, wide detuning, and twangy short strings.',
    voicing: {
      hammerHardness: 0.70,
      unisonDetune: 2.50,
      stretchTuning: 0.20,
      stringLengthScale: 0.65,
      a4FreqHz: 435.0,
    },
    space: {
      bodyId: BODY_VINTAGE_SALOON,
      perspective: PERSPECTIVE_PLAYER, // 0
      lidState: LID_HALF,              // 1
      reverbDecayS: 0.55,
      reverbMix: 0.08,
    },
  },
];

export const DEFAULT_PRESET_ID = 'concert-grand';

const PRESET_LOOKUP = new Map<string, ParameterPreset>(
  PIANO_PRESETS.map((p) => [p.id, p]),
);

export function getPreset(id: string): ParameterPreset | undefined {
  return PRESET_LOOKUP.get(id);
}
