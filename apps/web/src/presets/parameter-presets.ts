/** Physical modeling parameter presets for PianoSim.
 * Encapsulates macroscopic Voicing and Acoustic Space configurations into distinct piano archetypes.
 */

import {
  LID_CLOSED,
  LID_HALF,
  LID_OPEN,
  PERSPECTIVE_CLOSE,
  PERSPECTIVE_PLAYER,
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
    readonly perspective: ListenerPerspective; // ListenerPerspective (0: Player, 1: Close, 2: Hall, 3: Binaural)
    readonly lidState: LidState;                // LidState (0: Closed, 1: Half, 2: Open)
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
      perspective: PERSPECTIVE_PLAYER, // 0
      lidState: LID_OPEN,              // 2
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
      perspective: PERSPECTIVE_CLOSE,  // 1
      lidState: LID_CLOSED,            // 0
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
      perspective: PERSPECTIVE_CLOSE,  // 1
      lidState: LID_OPEN,              // 2
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
      perspective: PERSPECTIVE_PLAYER, // 0
      lidState: LID_HALF,              // 1
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
