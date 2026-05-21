await Bun.build({
    entrypoints: ['index.html'],
    outdir: './build',
    splitting: false,
    minify: {
        whitespace: false,
        identifiers: true,
        syntax: true,
    },
    sourcemap: "none",
});