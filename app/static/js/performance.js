"use strict";
import * as d3 from "https:/cdn.skypack.dev/d3@7";
import {scaling, colorBy, firstN, padding, width} from "./config.js";
import {dump, data} from "./data.js";

const columns = d3.range(firstN)
const size = (width - (columns.length + 1) * padding) / columns.length + padding

const N = Math.floor(dump["eigen"].length / 50)
const payoffs = columns.map(c => data.map(d1 => data.map(d2 => d1.embedding[2*c] * d2.embedding[2*c+1] - d2.embedding[2*c] * d1.embedding[2*c+1]))) 

// `recovered` should be similar to `dump["payoff"]`
// const recovered = payoffs.reduce((a, b) => a.map((row, i) => row.map((col, j) => col + b[i][j])))

const x = columns.map(c => d3.scaleBand().domain([0, N - 1]).padding(0.05))
const y = columns.map(c => d3.scaleBand().domain([0, N - 1]).padding(0.05))
const z = columns.map(c => d3.scaleLinear()
    .domain(d3.extent(payoffs[c]))
    .range(["lightblue", "blue"]))

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

cell.each(
    c => d3.select(this).selectAll("circle")
    	.data(d3.range(N * N).map(ij => payoffs[c][Math.floor(ij / N)][ij % N]))
        .join("circle")
        .attr("cx", (_, ij) => x[c](Math.floor(ij / N)))
        .attr("cy", (_, ij) => y[c](ij % N))
        .attr("r", 3.5)
        .style("fill", z[c])
        .style("opacity", 0.8))

const circle = cell.selectAll("circle");

cell.call(x => x, circle, svg);

svg.property("value", [])

export const performance_svg = svg;
