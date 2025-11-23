#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const https = require("https");

/**
 * Convert Mermaid diagram to SVG using Mermaid Ink API
 * Usage: node mermaid-to-svg.js input.mermaid [output.svg]
 */

function convertMermaidToSvg(inputFile, outputFile) {
  // Check if input file exists
  if (!fs.existsSync(inputFile)) {
    console.error(`Error: Input file '${inputFile}' not found`);
    process.exit(1);
  }

  // Read the mermaid file
  const mermaidContent = fs.readFileSync(inputFile, "utf8");

  // If no output file specified, use input filename with .svg extension
  if (!outputFile) {
    const baseName = path.basename(inputFile, path.extname(inputFile));
    outputFile = path.join(path.dirname(inputFile), `${baseName}.svg`);
  }

  // Encode mermaid content to base64 (URL-safe)
  const base64String = Buffer.from(mermaidContent, "utf8")
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");

  // Fetch SVG from Mermaid Ink API
  const url = `https://mermaid.ink/svg/${base64String}`;

  console.log("Fetching SVG from Mermaid Ink API...");

  https
    .get(url, (response) => {
      if (response.statusCode !== 200) {
        console.error(`Error: API returned status code ${response.statusCode}`);
        process.exit(1);
      }

      let svgData = "";

      response.on("data", (chunk) => {
        svgData += chunk;
      });

      response.on("end", () => {
        // Save SVG to file
        fs.writeFileSync(outputFile, svgData);
        console.log(`✓ SVG exported successfully to: ${outputFile}`);
      });
    })
    .on("error", (error) => {
      console.error("Error fetching SVG:", error.message);
      process.exit(1);
    });
}

// Parse command line arguments
const args = process.argv.slice(2);

if (args.length === 0) {
  console.log("Usage: node mermaid-to-svg.js <input.mermaid> [output.svg]");
  console.log("\nExample:");
  console.log("  node mermaid-to-svg.js diagram.mermaid");
  console.log("  node mermaid-to-svg.js diagram.mermaid output.svg");
  console.log("\nThis script uses the Mermaid Ink API (https://mermaid.ink/)");
  console.log("No dependencies required!");
  process.exit(1);
}

const inputFile = args[0];
const outputFile = args[1];

convertMermaidToSvg(inputFile, outputFile);
