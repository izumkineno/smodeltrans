import { describe, expect, test } from "bun:test";
import { buildOcrSelectionModel, resolveOcrSelectionRange } from "./ocr-selection";
import type { OcrRegion } from "./translation-provider";

const quad = (left: number, right: number): OcrRegion["quad"] => [
  [left, 0],
  [right, 0],
  [right, 10],
  [left, 10],
];

describe("OCR selection model", () => {
  test("orders regions and character boxes while preserving native text offsets", () => {
    const regions: OcrRegion[] = [
      {
        order: 2,
        quad: quad(0, 10),
        recognizedText: "你",
        charBoxes: [{ order: 1, quad: quad(0, 10), recognizedText: "你" }],
      },
      {
        order: 1,
        quad: quad(0, 20),
        recognizedText: "AB",
        charBoxes: [
          { order: 2, quad: quad(10, 20), recognizedText: "B" },
          { order: 1, quad: quad(0, 10), recognizedText: "A" },
        ],
      },
    ];

    const model = buildOcrSelectionModel(regions);

    expect(model.text).toBe("AB\n你");
    expect(model.characters.map((character) => character.text)).toEqual(["A", "B", "你"]);
    expect(model.characters.map(({ start, end }) => [start, end])).toEqual([
      [0, 1],
      [1, 2],
      [3, 4],
    ]);
    expect(resolveOcrSelectionRange(model, 2, 0)).toEqual({
      firstIndex: 0,
      lastIndex: 2,
      start: 0,
      end: 4,
      text: "AB\n你",
    });
  });

  test("falls back to a selectable region when character boxes are unavailable", () => {
    const model = buildOcrSelectionModel([
      {
        order: 1,
        quad: quad(3, 30),
        recognizedText: "fallback",
        charBoxes: [],
      },
    ]);

    expect(model.characters).toHaveLength(1);
    expect(model.characters[0].text).toBe("fallback");
    expect(model.characters[0].quad).toEqual(quad(3, 30));
    expect(resolveOcrSelectionRange(model, -1, 0)).toBeNull();
  });
});
