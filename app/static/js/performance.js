"use strict";
import * as d3 from "https://cdn.skypack.dev/d3@7";
import {colorBy, firstN, padding} from "./config.js";
import {dump, data} from "./data.js";

const w = window.innerWidth
const h = window.innerHeight
const width = 1.25 * w
const size = 0.28 * h

const columns = d3.range(firstN)

const N = Math.floor(dump["eigen"].length / 5)
const payoffs = columns.map(c => data.map(d1 => data.map(d2 => d1.embedding[2*c] * d2.embedding[2*c+1] - d2.embedding[2*c] * d1.embedding[2*c+1]))) 

const feature_in_interest = data.map(d => d.feature[colorBy])
const sorted_index = Array.from(Array(feature_in_interest.length).keys())
  .sort((a, b) => feature_in_interest[a] < feature_in_interest[b] ? -1 : (feature_in_interest[b] < feature_in_interest[a]) | 0)
const payoffs_flat = columns.map(c => d3.range(N * N).map(ij => payoffs[c][sorted_index[Math.floor(ij / N)]][sorted_index[ij % N]]))

const x = columns.map(c => d3.scaleLinear().domain([0, N - 1]).rangeRound([padding / 2, size - padding / 2]))
const y = columns.map(c => d3.scaleLinear().domain([0, N - 1]).rangeRound([padding / 2, size - padding / 2]))
const z = columns.map(c => d3.scaleLinear()
    .domain(d3.extent(payoffs_flat[c]))
    .range(["black", "white"]))

const svg = d3.create("svg")
    .attr("viewBox", [-padding, 0, width, width]);

const cell = svg.append("g")
    .selectAll("g")
    .data(d3.range(firstN))
    .join("g")
    .attr("transform", (i) => `translate(${i * size}, 0)`);

cell.append("rect")
    .attr("fill", "none")
    .attr("stroke", "#aaa")
    .attr("x", padding / 2 + 0.5)
    .attr("y", padding / 2 + 0.5)
    .attr("width", size - padding)
    .attr("height", size - padding);

cell.each(function (c) {
    d3.select(this).selectAll("circle")
        .data(payoffs_flat[c])
        .join("circle")
        .attr("cx", (_, ij) => x[c](Math.floor(ij / N)))
        .attr("cy", (_, ij) => y[c](ij % N))
        .attr("r", 0.8)
        .attr("fill", z[c]);
});

svg.property("value", [])

export const performance_svg = svg;
