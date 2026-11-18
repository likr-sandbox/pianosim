/** Curated classical demo score registry backed by canonical binary SMF (.mid) files. */

import { parseSmf, type ParsedSmf } from './smf-parser';

export interface DemoScoreDescriptor {
  readonly id: string;
  readonly title: string;
  readonly composer: string;
  readonly era: string;
  readonly description: string;
  readonly durationSeconds: number;
  readonly fileName: string;
}

export const DEMO_SCORES: ReadonlyArray<DemoScoreDescriptor> = [
  {
    id: 'bach-prelude',
    title: 'Bach: Prelude in C (BWV 846)',
    composer: 'J.S. Bach',
    era: 'Baroque (1722)',
    description: 'Crisp unpedaled 16th-note arpeggios, Railsback inharmonicity, polyphonic clarity.',
    durationSeconds: 33.0,
    fileName: '01-bach-prelude-bwv846.mid',
  },
  {
    id: 'chopin-nocturne',
    title: 'Chopin: Nocturne in Eb (Op. 9 No. 2)',
    composer: 'F. Chopin',
    era: 'Romantic (1832)',
    description: 'Bel canto singing melody, wide bass leaps, rhythmic damper pedal catch-and-release.',
    durationSeconds: 28.0,
    fileName: '02-chopin-nocturne-op9-no2.mid',
  },
  {
    id: 'beethoven-moonlight',
    title: 'Beethoven: Moonlight Sonata (Op. 27 No. 2)',
    composer: 'L. v. Beethoven',
    era: 'Classical (1801)',
    description: 'Full damper pedal wash, deep C# minor bass octaves, sympathetic bridge resonance.',
    durationSeconds: 38.0,
    fileName: '03-beethoven-moonlight-op27-no2.mid',
  },
  {
    id: 'debussy-claire',
    title: 'Debussy: Clair de Lune',
    composer: 'C. Debussy',
    era: 'Impressionism (1905)',
    description: 'Delicate pianissimo touch, high-register shimmer, Una Corda soft pedal trichord shift.',
    durationSeconds: 35.0,
    fileName: '04-debussy-clair-de-lune.mid',
  },
  {
    id: 'satie-gymnopedie',
    title: 'Satie: Gymnopédie No. 1',
    composer: 'E. Satie',
    era: 'Avant-Garde (1888)',
    description: 'Spacious ambient transparency, two-stage string damping, acoustic room reflection.',
    durationSeconds: 36.0,
    fileName: '05-satie-gymnopedie-no1.mid',
  },
  {
    id: 'rachmaninoff-prelude',
    title: 'Rachmaninoff: Prelude in C# Minor (Op. 3 No. 2)',
    composer: 'S. Rachmaninoff',
    era: 'Late Romantic (1892)',
    description: 'Fortissimo dynamic stress, hammer felt compression hardening, Zone 9 keybed thump.',
    durationSeconds: 25.0,
    fileName: '06-rachmaninoff-prelude-op3-no2.mid',
  },
];

const SCORE_CACHE = new Map<string, ParsedSmf>();
const IN_FLIGHT = new Map<string, Promise<ParsedSmf>>();

export function getDemoMidiUrl(fileName: string): string {
  const base = (import.meta as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? '/';
  return `${base}demos/${fileName}`;
}

/** Fetch and parse a canonical `.mid` demo score by ID with per-ID caching. */
export async function fetchDemoScore(id: string): Promise<ParsedSmf> {
  const desc = DEMO_SCORES.find((s) => s.id === id) ?? DEMO_SCORES[0];
  const cached = SCORE_CACHE.get(desc.id);
  if (cached) {
    return cached;
  }
  const existing = IN_FLIGHT.get(desc.id);
  if (existing) {
    return existing;
  }

  const req = (async () => {
    const res = await fetch(getDemoMidiUrl(desc.fileName));
    if (!res.ok) {
      throw new Error(`Failed to fetch demo MIDI ${desc.fileName}: HTTP ${res.status}`);
    }
    const buf = await res.arrayBuffer();
    const parsed = parseSmf(buf);
    SCORE_CACHE.set(desc.id, parsed);
    return parsed;
  })();

  IN_FLIGHT.set(desc.id, req);
  try {
    return await req;
  } finally {
    IN_FLIGHT.delete(desc.id);
  }
}
