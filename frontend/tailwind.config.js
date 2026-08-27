/** @type {import('tailwindcss').Config} */

/* Every scale below points at a custom property defined in src/theme/tokens.css.
 * Nothing here restates a literal value, so the token file stays the only place
 * a colour or size is actually decided — and runtime-computed values (grade
 * colours, score-bar widths) can flow through the same names as inline
 * `style={{ '--x': … }}` without a second vocabulary. */

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    // Replace, not extend: the defaults are what produced 15 font sizes and 13
    // radii. If a value isn't in the token file, it shouldn't be reachable.
    colors: {
      transparent: "transparent",
      current: "currentColor",

      well:         "var(--well)",
      ground:       "var(--ground)",
      surface:      "var(--surface)",
      raised:       "var(--raised)",
      "raised-hover": "var(--raised-hover)",
      line:         "var(--line)",
      "line-strong": "var(--line-strong)",

      ink:     "var(--ink)",
      "ink-2": "var(--ink-2)",
      "ink-3": "var(--ink-3)",
      "ink-4": "var(--ink-4)",

      "grade-strong":  "var(--grade-strong)",
      "grade-weak":    "var(--grade-weak)",
      "grade-pending": "var(--grade-pending)",

      mark:       "var(--mark)",
      "mark-ink": "var(--mark-ink)",
      "mark-dim": "var(--mark-dim)",

      focus:         "var(--focus)",
      "focus-inset": "var(--focus-inset)",
      scrim:         "var(--scrim)",

      // Alarm states. Deliberately NOT in the general palette: status chrome is
      // neutral until something actually needs the user, so these two are the
      // only non-neutral, non-mark colours the chrome may ever show.
      "alarm-warn": "var(--alarm-warn)",
      "alarm-crit": "var(--alarm-crit)",

      // Machine voice — the one cold accent, reserved for AI output.
      ai: "var(--ai)",
      "ai-dim": "var(--ai-dim)",
      "ai-ink": "var(--ai-ink)",
    },

    fontFamily: {
      sans: "var(--font-sans)",
      mono: "var(--font-mono)",
    },

    fontSize: {
      xs: ["var(--text-xs)", { lineHeight: "var(--leading-body)" }],
      sm: ["var(--text-sm)", { lineHeight: "var(--leading-body)" }],
      md: ["var(--text-md)", { lineHeight: "var(--leading-body)" }],
      lg: ["var(--text-lg)", { lineHeight: "var(--leading-display)" }],
      xl: ["var(--text-xl)", { lineHeight: "var(--leading-display)" }],
    },

    // Leading utilities resolve to the SAME tokens the inline styles use.
    // Without this the class layer fell through to Tailwind's stock scale, so
    // `leading-relaxed` (1.625) sat beside `--leading-prose` (1.6) doing the
    // same job by a different number — the exact split the token guard exists
    // to prevent, just one layer up. Replacing the key (rather than extending)
    // also retires the stock names, so there is one way to say this.
    lineHeight: {
      none: "var(--leading-none)",
      body: "var(--leading-body)",
      prose: "var(--leading-prose)",
      display: "var(--leading-display)",
    },

    borderRadius: {
      none: "0",
      // The contact-sheet cell only — see the --r-cell note in tokens.css.
      cell: "var(--r-cell)",
      sm: "var(--r-sm)",
      md: "var(--r-md)",
      lg: "var(--r-lg)",
      xl: "var(--r-xl)",
      full: "9999px",
    },

    spacing: {
      0: "0",
      1: "var(--sp-1)",
      2: "var(--sp-2)",
      3: "var(--sp-3)",
      4: "var(--sp-4)",
      6: "var(--sp-6)",
      8: "var(--sp-8)",
      12: "var(--sp-12)",
      px: "1px",
      full: "100%",
    },

    transitionTimingFunction: { DEFAULT: "var(--ease)", ease: "var(--ease)", spring: "var(--spring)" },
    transitionDuration: { DEFAULT: "var(--t-fast)", fast: "var(--t-fast)", slow: "var(--t-slow)", spring: "var(--t-spring)" },

    extend: {
      // Structural dimensions, named. Keeps w-72 / h-16 magic numbers out of
      // the markup now that the numeric spacing scale is deliberately sparse.
      width:  { panel: "var(--w-panel)", sidebar: "var(--w-sidebar)", stage: "var(--w-stage)",
                palette: "var(--w-palette)", kbd: "var(--w-kbd)" },
      height: { thumb: "var(--h-thumb)", toolbar: "var(--h-toolbar)", rule: "var(--h-rule)", palette: "var(--h-palette)",
                field: "var(--h-field)", row: "var(--h-row)" },
      minWidth: { kbd: "var(--w-kbd)" },
      minHeight: { toolbar: "var(--h-toolbar)" },
      maxHeight: { palette: "var(--h-palette)" },
      borderWidth: { DEFAULT: "1px", 0: "0", 2: "2px" },
      opacity: { reject: "var(--dim-reject)" },
      // No `keyframes` block here on purpose. Every @keyframes lives in
      // src/index.css, because Tailwind emits keyframes ONLY for animate-*
      // utilities it can see in the source — an inline `animation: spin` is
      // invisible to that scan. Defining them here made 11 spinners and every
      // skeleton silently stop moving. These entries just name the animation;
      // the CSS file supplies it. `npm run lint:tokens` enforces the pairing.
      animation: {
        shimmer: "shimmer 1.5s linear infinite",
        "fade-in": "fadeIn var(--t-slow) var(--ease)",
        "dialog-in": "dialogIn var(--t-slow) var(--ease)",
        sweep: "sweep 1.2s ease-in-out infinite",
        "chip-in": "chipIn var(--t-spring) var(--spring) both",
        "pop-in": "popIn var(--t-spring) var(--spring) both",
        "palette-in": "paletteIn var(--t-spring) var(--spring) both",
      },
    },
  },
  plugins: [],
};
