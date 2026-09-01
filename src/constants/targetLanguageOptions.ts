export const TARGET_LANGUAGE_OPTIONS: Array<{ label: string; value: string }> = [
  { label: "中文 (Chinese · zh)", value: "Chinese" },
  { label: "英语 (English · en)", value: "English" },
  { label: "法语 (French · fr)", value: "French" },
  { label: "葡萄牙语 (Portuguese · pt)", value: "Portuguese" },
  { label: "西班牙语 (Spanish · es)", value: "Spanish" },
  { label: "日语 (Japanese · ja)", value: "Japanese" },
  { label: "土耳其语 (Turkish · tr)", value: "Turkish" },
  { label: "俄语 (Russian · ru)", value: "Russian" },
  { label: "阿拉伯语 (Arabic · ar)", value: "Arabic" },
  { label: "韩语 (Korean · ko)", value: "Korean" },
  { label: "泰语 (Thai · th)", value: "Thai" },
  { label: "意大利语 (Italian · it)", value: "Italian" },
  { label: "德语 (German · de)", value: "German" },
  { label: "越南语 (Vietnamese · vi)", value: "Vietnamese" },
  { label: "马来语 (Malay · ms)", value: "Malay" },
  { label: "印尼语 (Indonesian · id)", value: "Indonesian" },
  { label: "菲律宾语 (Filipino · tl)", value: "Filipino" },
  { label: "印地语 (Hindi · hi)", value: "Hindi" },
  { label: "繁体中文 (Traditional Chinese · zh-Hant)", value: "Traditional Chinese" },
  { label: "波兰语 (Polish · pl)", value: "Polish" },
  { label: "捷克语 (Czech · cs)", value: "Czech" },
  { label: "荷兰语 (Dutch · nl)", value: "Dutch" },
  { label: "高棉语 (Khmer · km)", value: "Khmer" },
  { label: "缅甸语 (Burmese · my)", value: "Burmese" },
  { label: "波斯语 (Persian · fa)", value: "Persian" },
  { label: "古吉拉特语 (Gujarati · gu)", value: "Gujarati" },
  { label: "乌尔都语 (Urdu · ur)", value: "Urdu" },
  { label: "泰卢固语 (Telugu · te)", value: "Telugu" },
  { label: "马拉地语 (Marathi · mr)", value: "Marathi" },
  { label: "希伯来语 (Hebrew · he)", value: "Hebrew" },
  { label: "孟加拉语 (Bengali · bn)", value: "Bengali" },
  { label: "泰米尔语 (Tamil · ta)", value: "Tamil" },
  { label: "乌克兰语 (Ukrainian · uk)", value: "Ukrainian" },
  { label: "藏语 (Tibetan · bo)", value: "Tibetan" },
  { label: "哈萨克语 (Kazakh · kk)", value: "Kazakh" },
  { label: "蒙古语 (Mongolian · mn)", value: "Mongolian" },
  { label: "维吾尔语 (Uyghur · ug)", value: "Uyghur" },
  { label: "粤语 (Cantonese · yue)", value: "Cantonese" },
];

export const SUPPORTED_TARGET_LANGUAGE_VALUES: Record<string, true> = Object.fromEntries(
  TARGET_LANGUAGE_OPTIONS.map((o) => [o.value, true as const]),
) as Record<string, true>;

export function isSupportedTargetLanguage(value: string): boolean {
  return Boolean(SUPPORTED_TARGET_LANGUAGE_VALUES[value.trim()]);
}
