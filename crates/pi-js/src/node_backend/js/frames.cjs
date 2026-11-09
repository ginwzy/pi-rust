"use strict";

module.exports = function readFrames(receive) {
  // Scan each chunk once; join an incomplete frame only after its newline arrives.
  let parts = [];
  return (chunk) => {
    let start = 0;
    let newline;
    while ((newline = chunk.indexOf("\n", start)) >= 0) {
      let line = chunk.slice(start, newline);
      if (parts.length) {
        parts.push(line);
        line = parts.join("");
        parts = [];
      }
      start = newline + 1;
      if (line.trim()) receive(JSON.parse(line));
    }
    if (start < chunk.length) parts.push(chunk.slice(start));
  };
};
