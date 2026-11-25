// Intl: the subset extensions and pi-tui use. Segmenter follows Unicode segmentation (UAX #29);
// NumberFormat, DateTimeFormat, PluralRules, Collator, and DisplayNames implement en-US behavior only.
const encoding = native.encoding;
const WORD_LIKE = /[\p{L}\p{N}]/u;

function segmentsOf(input, granularity) {
  const text = String(input);
  const parts = encoding.segment(text, granularity);
  const segments = new Array(parts.length);
  let index = 0;
  for (let i = 0; i < parts.length; i++) {
    const segment = { segment: parts[i], index, input: text };
    if (granularity === "word") segment.isWordLike = WORD_LIKE.test(parts[i]);
    segments[i] = segment;
    index += parts[i].length;
  }
  return segments;
}

class Segments {
  #list;
  constructor(list) {
    this.#list = list;
  }
  containing(codeUnitIndex = 0) {
    const index = Number(codeUnitIndex);
    return this.#list.find((segment) => index >= segment.index && index < segment.index + segment.segment.length);
  }
  [Symbol.iterator]() {
    return this.#list[Symbol.iterator]();
  }
}

class Segmenter {
  #granularity;
  constructor(_locales, options = {}) {
    const granularity = options.granularity ?? "grapheme";
    if (!["grapheme", "word", "sentence"].includes(granularity)) {
      throw new RangeError(`Value ${granularity} out of range for Intl.Segmenter options property granularity`);
    }
    this.#granularity = granularity;
  }
  segment(input) {
    return new Segments(segmentsOf(input, this.#granularity));
  }
  resolvedOptions() {
    return { locale: "en-US", granularity: this.#granularity };
  }
  static supportedLocalesOf() {
    return ["en-US"];
  }
}

function group(digits) {
  return digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

class NumberFormat {
  #options;
  constructor(_locales, options = {}) {
    this.#options = { ...options };
  }
  format(value) {
    const options = this.#options;
    let number = Number(value);
    if (!Number.isFinite(number)) return String(number);
    const style = options.style ?? "decimal";
    if (style === "percent") number *= 100;
    const defaultMax = { currency: 2, percent: 0 }[style] ?? 3;
    const min = options.minimumFractionDigits ?? (style === "currency" ? 2 : 0);
    const max = Math.max(min, options.maximumFractionDigits ?? defaultMax);
    let text = Math.abs(number).toFixed(max);
    if (text.includes(".")) {
      let [whole, fraction] = text.split(".");
      while (fraction.length > min && fraction.endsWith("0")) fraction = fraction.slice(0, -1);
      text = fraction.length > 0 ? `${whole}.${fraction}` : whole;
    }
    const [whole, fraction] = text.split(".");
    text = (options.useGrouping === false ? whole : group(whole)) + (fraction ? `.${fraction}` : "");
    const sign = number < 0 && Number(text.replace(/,/g, "")) !== 0 ? "-" : "";
    if (style === "percent") return `${sign}${text}%`;
    if (style === "currency") {
      const symbol = !options.currency || options.currency === "USD" ? "$" : `${options.currency} `;
      return `${sign}${symbol}${text}`;
    }
    if (style === "unit" && options.unit) return `${sign}${text} ${options.unit}`;
    return sign + text;
  }
  formatToParts(value) {
    return [{ type: "literal", value: this.format(value) }];
  }
  resolvedOptions() {
    return { locale: "en-US", numberingSystem: "latn", style: "decimal", ...this.#options };
  }
  static supportedLocalesOf() {
    return ["en-US"];
  }
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
  "November", "December"];
const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

class DateTimeFormat {
  #options;
  constructor(_locales, options = {}) {
    this.#options = { ...options };
  }
  format(value) {
    const date = value === undefined ? new Date() : new Date(value);
    const options = this.#options;
    const hasDate = options.year || options.month || options.day || options.weekday || options.dateStyle;
    const hasTime = options.hour || options.minute || options.second || options.timeStyle;
    if (!hasDate && !hasTime) return date.toLocaleDateString();
    const name = (names, index, width) => (width === "long" ? names[index] : names[index].slice(0, 3));
    const parts = [];
    if (hasDate) {
      if (options.weekday) parts.push(name(WEEKDAYS, date.getDay(), options.weekday));
      if (options.month === "long" || options.month === "short") {
        const day = options.day ? ` ${date.getDate()}` : "";
        const year = options.year ? `, ${date.getFullYear()}` : "";
        parts.push(`${name(MONTHS, date.getMonth(), options.month)}${day}${year}`);
      } else {
        parts.push(date.toLocaleDateString());
      }
    }
    if (hasTime) parts.push(date.toLocaleTimeString());
    return parts.join(", ");
  }
  formatToParts(value) {
    return [{ type: "literal", value: this.format(value) }];
  }
  resolvedOptions() {
    const timeZone = process.env.TZ || "UTC";
    return { locale: "en-US", calendar: "gregory", numberingSystem: "latn", timeZone, ...this.#options };
  }
  static supportedLocalesOf() {
    return ["en-US"];
  }
}

class PluralRules {
  select(value) {
    return Number(value) === 1 ? "one" : "other";
  }
  resolvedOptions() {
    return { locale: "en-US", type: "cardinal" };
  }
}

class Collator {
  #options;
  constructor(_locales, options = {}) {
    this.#options = options;
  }
  compare = (left, right) => {
    const caseless = this.#options.sensitivity === "base" || this.#options.sensitivity === "accent";
    const normalize = (value) => (caseless ? String(value).toLowerCase() : String(value));
    const split = (value) => (this.#options.numeric ? value.split(/(\d+)/) : [value]);
    const leftParts = split(normalize(left));
    const rightParts = split(normalize(right));
    for (let i = 0; i < Math.max(leftParts.length, rightParts.length); i++) {
      const a = leftParts[i] ?? "";
      const b = rightParts[i] ?? "";
      const order = i % 2 === 1 ? Number(a) - Number(b) : a < b ? -1 : Number(a > b);
      if (order !== 0) return Math.sign(order);
    }
    return 0;
  };
  resolvedOptions() {
    return { locale: "en-US", usage: "sort", sensitivity: "variant", ...this.#options };
  }
}

const DISPLAY_TYPES = ["language", "region", "script", "currency", "calendar", "dateTimeField"];
const DISPLAY_STYLES = ["long", "short", "narrow"];
const DISPLAY_FALLBACKS = ["code", "none"];
const DISPLAY_LOCALE_MATCHERS = ["lookup", "best fit"];
const LANGUAGE_DISPLAYS = ["dialect", "standard"];

const REGION_RE = /^[A-Za-z]{2}$|^\d{3}$/;
const SCRIPT_RE = /^[A-Za-z]{4}$/;
const LANGUAGE_RE = /^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$/;

const LANGUAGE_NAMES = {
  aa: "Afar", ab: "Abkhazian", ae: "Avestan", af: "Afrikaans", ak: "Akan", am: "Amharic", an: "Aragonese",
  ar: "Arabic", as: "Assamese", av: "Avaric", ay: "Aymara", az: "Azerbaijani", ba: "Bashkir", be: "Belarusian",
  bg: "Bulgarian", bh: "Bhojpuri", bi: "Bislama", bm: "Bambara", bn: "Bangla", bo: "Tibetan", br: "Breton",
  bs: "Bosnian", ca: "Catalan", ce: "Chechen", ch: "Chamorro", co: "Corsican", cr: "Cree", cs: "Czech",
  cu: "Church Slavic", cv: "Chuvash", cy: "Welsh", da: "Danish", de: "German", dv: "Divehi", dz: "Dzongkha",
  ee: "Ewe", el: "Greek", en: "English", eo: "Esperanto", es: "Spanish", et: "Estonian", eu: "Basque",
  fa: "Persian", ff: "Fula", fi: "Finnish", fj: "Fijian", fo: "Faroese", fr: "French", fy: "Western Frisian",
  ga: "Irish", gd: "Scottish Gaelic", gl: "Galician", gn: "Guarani", gu: "Gujarati", gv: "Manx", ha: "Hausa",
  he: "Hebrew", hi: "Hindi", ho: "Hiri Motu", hr: "Croatian", ht: "Haitian Creole", hu: "Hungarian",
  hy: "Armenian", hz: "Herero", ia: "Interlingua", id: "Indonesian", ie: "Interlingue", ig: "Igbo",
  ii: "Sichuan Yi", ik: "Inupiaq", in: "Indonesian", io: "Ido", is: "Icelandic", it: "Italian", iu: "Inuktitut",
  iw: "Hebrew", ja: "Japanese", ji: "Yiddish", jv: "Javanese", jw: "Javanese", ka: "Georgian", kg: "Kongo",
  ki: "Kikuyu", kj: "Kuanyama", kk: "Kazakh", kl: "Kalaallisut", km: "Khmer", kn: "Kannada", ko: "Korean",
  kr: "Kanuri", ks: "Kashmiri", ku: "Kurdish", kv: "Komi", kw: "Cornish", ky: "Kyrgyz", la: "Latin",
  lb: "Luxembourgish", lg: "Ganda", li: "Limburgish", ln: "Lingala", lo: "Lao", lt: "Lithuanian",
  lu: "Luba-Katanga", lv: "Latvian", mg: "Malagasy", mh: "Marshallese", mi: "Māori", mk: "Macedonian",
  ml: "Malayalam", mn: "Mongolian", mo: "Romanian", mr: "Marathi", ms: "Malay", mt: "Maltese", my: "Burmese",
  na: "Nauru", nb: "Norwegian Bokmål", nd: "North Ndebele", ne: "Nepali", ng: "Ndonga", nl: "Dutch",
  nn: "Norwegian Nynorsk", no: "Norwegian", nr: "South Ndebele", nv: "Navajo", ny: "Nyanja", oc: "Occitan",
  oj: "Ojibwa", om: "Oromo", or: "Odia", os: "Ossetic", pa: "Punjabi", pi: "Pali", pl: "Polish", ps: "Pashto",
  pt: "Portuguese", qu: "Quechua", rm: "Romansh", rn: "Rundi", ro: "Romanian", ru: "Russian", rw: "Kinyarwanda",
  sa: "Sanskrit", sc: "Sardinian", sd: "Sindhi", se: "Northern Sami", sg: "Sango", sh: "Serbian (Latin)",
  si: "Sinhala", sk: "Slovak", sl: "Slovenian", sm: "Samoan", sn: "Shona", so: "Somali", sq: "Albanian",
  sr: "Serbian", ss: "Swati", st: "Southern Sotho", su: "Sundanese", sv: "Swedish", sw: "Swahili", ta: "Tamil",
  te: "Telugu", tg: "Tajik", th: "Thai", ti: "Tigrinya", tk: "Turkmen", tl: "Filipino", tn: "Tswana",
  to: "Tongan", tr: "Turkish", ts: "Tsonga", tt: "Tatar", tw: "Akan", ty: "Tahitian", ug: "Uyghur",
  uk: "Ukrainian", ur: "Urdu", uz: "Uzbek", ve: "Venda", vi: "Vietnamese", vo: "Volapük", wa: "Walloon",
  wo: "Wolof", xh: "Xhosa", yi: "Yiddish", yo: "Yoruba", za: "Zhuang", zh: "Chinese", zu: "Zulu",
};

const REGION_NAMES = {
  AU: "Australia", BR: "Brazil", CA: "Canada", CH: "Switzerland", CN: "China", DE: "Germany", ES: "Spain",
  FR: "France", GB: "United Kingdom", HK: "Hong Kong SAR China", IN: "India", IT: "Italy", JP: "Japan",
  KR: "South Korea", MX: "Mexico", NL: "Netherlands", RU: "Russia", SE: "Sweden", SG: "Singapore", TW: "Taiwan",
  US: "United States", ZZ: "Unknown Region",
};

const SCRIPT_NAMES = {
  Arab: "Arabic", Cyrl: "Cyrillic", Deva: "Devanagari", Grek: "Greek", Hang: "Hangul", Hans: "Simplified",
  Hant: "Traditional", Hebr: "Hebrew", Jpan: "Japanese", Kore: "Korean", Latn: "Latin", Thai: "Thai",
};

const CURRENCY_NAMES = {
  AUD: "Australian Dollar", CAD: "Canadian Dollar", CHF: "Swiss Franc", CNY: "Chinese Yuan", EUR: "Euro",
  GBP: "British Pound", HKD: "Hong Kong Dollar", INR: "Indian Rupee", JPY: "Japanese Yen", KRW: "South Korean Won",
  USD: "US Dollar",
};

const CALENDAR_NAMES = {
  buddhist: "Buddhist Calendar", chinese: "Chinese Calendar", gregory: "Gregorian Calendar",
  hebrew: "Hebrew Calendar", islamic: "Hijri Calendar", iso8601: "ISO-8601 Calendar", japanese: "Japanese Calendar",
};

const DATE_TIME_FIELD_NAMES = {
  era: "era", year: "year", quarter: "quarter", month: "month", weekOfYear: "week", weekday: "day of the week",
  day: "day", dayPeriod: "AM/PM", hour: "hour", minute: "minute", second: "second", timeZoneName: "time zone",
};

const own = (table, key) => (Object.hasOwn(table, key) ? table[key] : undefined);
const titleCase = (value) => value[0].toUpperCase() + value.slice(1).toLowerCase();

const DISPLAY_TABLES = {
  region: { pattern: REGION_RE, normalize: (value) => value.toUpperCase(), names: REGION_NAMES },
  script: { pattern: SCRIPT_RE, normalize: titleCase, names: SCRIPT_NAMES },
  currency: { pattern: /^[A-Za-z]{3}$/, normalize: (value) => value.toUpperCase(), names: CURRENCY_NAMES },
  calendar: {
    pattern: /^[A-Za-z0-9]{3,8}(?:-[A-Za-z0-9]{3,8})*$/,
    normalize: (value) => value.toLowerCase(),
    names: CALENDAR_NAMES,
  },
};

function checkOption(property, value, allowed) {
  if (!allowed.includes(value)) {
    throw new RangeError(`Value ${value} out of range for Intl.DisplayNames options property ${property}`);
  }
  return value;
}

function lookupLanguage(code) {
  if (!LANGUAGE_RE.test(code)) throw new RangeError(`Invalid language argument: ${code}`);
  const parts = code.split("-");
  const primary = parts[0].toLowerCase();
  const base = own(LANGUAGE_NAMES, primary);
  if (!base) return [undefined, parts.length === 1 ? primary : code];
  if (parts.length === 1) return [base, primary];
  const qualifiers = parts.slice(1).map((part) => {
    if (SCRIPT_RE.test(part)) {
      const script = titleCase(part);
      return own(SCRIPT_NAMES, script) ?? script;
    }
    if (REGION_RE.test(part)) {
      const region = part.toUpperCase();
      return own(REGION_NAMES, region) ?? region;
    }
    return part;
  });
  return [`${base} (${qualifiers.join(", ")})`, code];
}

class DisplayNames {
  #type;
  #style;
  #fallback;
  #languageDisplay;
  constructor(_locales, options) {
    if (options === undefined || options === null || typeof options !== "object") {
      throw new TypeError("Intl.DisplayNames requires an options object");
    }
    checkOption("localeMatcher", options.localeMatcher ?? "best fit", DISPLAY_LOCALE_MATCHERS);
    this.#style = checkOption("style", options.style ?? "long", DISPLAY_STYLES);
    if (options.type === undefined) {
      throw new TypeError("Intl.DisplayNames options.type is required");
    }
    this.#type = checkOption("type", String(options.type), DISPLAY_TYPES);
    this.#fallback = checkOption("fallback", options.fallback ?? "code", DISPLAY_FALLBACKS);
    this.#languageDisplay = checkOption("languageDisplay", options.languageDisplay ?? "dialect", LANGUAGE_DISPLAYS);
  }
  of(code) {
    const raw = String(code);
    if (this.#type === "dateTimeField") {
      const field = own(DATE_TIME_FIELD_NAMES, raw);
      if (field === undefined) throw new RangeError(`Invalid dateTimeField argument: ${raw}`);
      return field;
    }
    let name;
    let normalized;
    if (this.#type === "language") {
      [name, normalized] = lookupLanguage(raw);
    } else {
      const table = DISPLAY_TABLES[this.#type];
      if (!table.pattern.test(raw)) throw new RangeError(`Invalid ${this.#type} argument: ${raw}`);
      normalized = table.normalize(raw);
      name = own(table.names, normalized);
    }
    if (name !== undefined) return name;
    return this.#fallback === "none" ? undefined : normalized;
  }
  resolvedOptions() {
    const resolved = { locale: "en-US", style: this.#style, type: this.#type, fallback: this.#fallback };
    if (this.#type === "language") resolved.languageDisplay = this.#languageDisplay;
    return resolved;
  }
  static supportedLocalesOf() {
    return ["en-US"];
  }
}

module.exports = {
  Segmenter,
  NumberFormat,
  DateTimeFormat,
  PluralRules,
  Collator,
  DisplayNames,
  getCanonicalLocales: (locales) => (locales === undefined ? [] : [].concat(locales)),
  supportedValuesOf: () => [],
};
