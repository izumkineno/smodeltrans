import type { OcrQuad, OcrRegion } from "./translation-provider";

export interface OcrSelectionCharacter {
  key: string;
  regionOrder: number;
  characterOrder: number;
  text: string;
  quad: OcrQuad;
  start: number;
  end: number;
}

export interface OcrSelectionModel {
  text: string;
  characters: OcrSelectionCharacter[];
}

export interface OcrSelectionRange {
  firstIndex: number;
  lastIndex: number;
  start: number;
  end: number;
  text: string;
}

export function buildOcrSelectionModel(regions: readonly OcrRegion[]): OcrSelectionModel {
  const orderedRegions = regions
    .map((region, sourceIndex) => ({ region, sourceIndex }))
    .sort((left, right) => left.region.order - right.region.order || left.sourceIndex - right.sourceIndex);
  const characters: OcrSelectionCharacter[] = [];
  let text = "";

  for (const [regionIndex, { region }] of orderedRegions.entries()) {
    if (regionIndex > 0) {
      text += "\n";
    }
    const boxes = region.charBoxes.length > 0
      ? region.charBoxes
          .map((box, sourceIndex) => ({ box, sourceIndex }))
          .sort((left, right) => left.box.order - right.box.order || left.sourceIndex - right.sourceIndex)
          .map(({ box }) => box)
      : [{ order: 1, quad: region.quad, recognizedText: region.recognizedText }];

    for (const [characterIndex, box] of boxes.entries()) {
      const start = text.length;
      text += box.recognizedText;
      characters.push({
        key: `${region.order}:${box.order}:${regionIndex}:${characterIndex}`,
        regionOrder: region.order,
        characterOrder: box.order,
        text: box.recognizedText,
        quad: box.quad,
        start,
        end: text.length,
      });
    }
  }

  return { text, characters };
}

export function resolveOcrSelectionRange(
  model: OcrSelectionModel,
  anchorIndex: number,
  focusIndex: number,
): OcrSelectionRange | null {
  if (
    anchorIndex < 0 ||
    focusIndex < 0 ||
    anchorIndex >= model.characters.length ||
    focusIndex >= model.characters.length
  ) {
    return null;
  }
  const firstIndex = Math.min(anchorIndex, focusIndex);
  const lastIndex = Math.max(anchorIndex, focusIndex);
  const start = model.characters[firstIndex].start;
  const end = model.characters[lastIndex].end;
  return {
    firstIndex,
    lastIndex,
    start,
    end,
    text: model.text.slice(start, end),
  };
}
