/** Curated classical demo score registry for PianoSim physical modeling showcases. */

import type { ParsedSmf } from './smf-parser';
import { createBachPrelude } from './scores/bach-prelude';
import { createBeethovenMoonlight } from './scores/beethoven-moonlight';
import { createChopinNocturne } from './scores/chopin-nocturne';
import { createDebussyClaire } from './scores/debussy-claire';
import { createRachmaninoffPrelude } from './scores/rachmaninoff-prelude';
import { createSatieGymnopedie } from './scores/satie-gymnopedie';
import type { DemoScoreDescriptor } from './scores/types';

export {
  createBachPrelude,
  createBeethovenMoonlight,
  createChopinNocturne,
  createDebussyClaire,
  createRachmaninoffPrelude,
  createSatieGymnopedie,
};
export type { DemoScoreDescriptor };

export const DEMO_SCORES: ReadonlyArray<DemoScoreDescriptor> = [
  {
    id: 'bach-prelude',
    title: 'Bach: Prelude in C (BWV 846)',
    composer: 'J.S. Bach',
    era: 'Baroque (1722)',
    description: 'Crisp unpedaled 16th-note arpeggios, Railsback inharmonicity, polyphonic clarity.',
    durationSeconds: 33.0,
    generator: createBachPrelude,
  },
  {
    id: 'chopin-nocturne',
    title: 'Chopin: Nocturne in Eb (Op. 9 No. 2)',
    composer: 'F. Chopin',
    era: 'Romantic (1832)',
    description: 'Bel canto singing melody, wide bass leaps, rhythmic damper pedal catch-and-release.',
    durationSeconds: 28.0,
    generator: createChopinNocturne,
  },
  {
    id: 'beethoven-moonlight',
    title: 'Beethoven: Moonlight Sonata (Op. 27 No. 2)',
    composer: 'L. v. Beethoven',
    era: 'Classical (1801)',
    description: 'Full damper pedal wash, deep C# minor bass octaves, sympathetic bridge resonance.',
    durationSeconds: 38.0,
    generator: createBeethovenMoonlight,
  },
  {
    id: 'debussy-claire',
    title: 'Debussy: Clair de Lune',
    composer: 'C. Debussy',
    era: 'Impressionism (1905)',
    description: 'Delicate pianissimo touch, high-register shimmer, Una Corda soft pedal trichord shift.',
    durationSeconds: 35.0,
    generator: createDebussyClaire,
  },
  {
    id: 'satie-gymnopedie',
    title: 'Satie: Gymnopédie No. 1',
    composer: 'E. Satie',
    era: 'Avant-Garde (1888)',
    description: 'Spacious ambient transparency, two-stage string damping, acoustic room reflection.',
    durationSeconds: 36.0,
    generator: createSatieGymnopedie,
  },
  {
    id: 'rachmaninoff-prelude',
    title: 'Rachmaninoff: Prelude in C# Minor (Op. 3 No. 2)',
    composer: 'S. Rachmaninoff',
    era: 'Late Romantic (1892)',
    description: 'Fortissimo dynamic stress, hammer felt compression hardening, Zone 9 keybed thump.',
    durationSeconds: 25.0,
    generator: createRachmaninoffPrelude,
  },
];

const SCORE_LOOKUP = new Map<string, () => ParsedSmf>(
  DEMO_SCORES.map((s) => [s.id, s.generator]),
);

/** Look up a demo score by ID; defaults to Bach Prelude if unknown for backward compatibility. */
export function getDemoScore(id: string): ParsedSmf {
  const gen = SCORE_LOOKUP.get(id);
  if (gen) {
    return gen();
  }
  return createBachPrelude();
}
