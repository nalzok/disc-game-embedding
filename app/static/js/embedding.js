"use strict";
import * as d3 from "https://cdn.skypack.dev/d3@7";
import {scaling, colorBy, firstN, padding, width} from "./config.js";
import {data} from "./data.js";

const columns = d3.range(firstN)
const size = (width - (columns.length + 1) * padding) / columns.length + padding

const x = scaling === "auto" ?
    columns.map(c => d3.scaleLinear()
        .domain(d3.extent(data, d => d.embedding[2 * c]).map(x => 1.2 * x))
        .rangeRound([padding / 2, size - padding / 2])) :
    columns.map(c => d3.scaleLinear()
        .domain((() => {
            const [lower, upper] = d3.extent(data, d => d.embedding[0]).map(x => 1.2 * x)
            const magnitude = Math.max(Math.abs(lower), Math.abs(upper))
            return [-magnitude, magnitude]
        })())
        .rangeRound([padding / 2, size - padding / 2]))

const y = scaling === "auto" ?
    columns.map(c => d3.scaleLinear()
        .domain(d3.extent(data, d => d.embedding[2 * c + 1]).map(x => 1.2 * x))
        .rangeRound([size - padding / 2, padding / 2])) :
    columns.map(c => d3.scaleLinear()
        .domain((() => {
            const [lower, upper] = d3.extent(data, d => d.embedding[0]).map(x => 1.2 * x)
            const magnitude = Math.max(Math.abs(lower), Math.abs(upper))
            return [-magnitude, magnitude]
        })())
        .rangeRound([size - padding / 2, padding / 2]))

const z = d3.scaleLinear()
    .domain(d3.extent(data, d => d.feature[colorBy]))
    .range(["lightblue", "blue"])

function brush(cell, circle, svg) {
    const brush = d3.brush()
        .extent([[padding / 2, padding / 2], [size - padding / 2, size - padding / 2]])
        .on("start", brushstarted)
        .on("brush", brushed)
        .on("end", brushended);

    cell.call(brush);

    let brushCell;

    // Clear the previously-active brush, if any.
    function brushstarted() {
        if (brushCell !== this) {
            d3.select(brushCell).call(brush.move, null);
            brushCell = this;
        }
    }

    // Highlight the selected circles.
    function brushed({selection}, i) {
        let selected = [];
        if (selection) {
            const [[x0, y0], [x1, y1]] = selection;
            const isSelected = d => x0 < x[i](d.embedding[2 * i]) && x1 > x[i](d.embedding[2 * i]) && y0 < y[i](d.embedding[2 * i + 1]) && y1 > y[i](d.embedding[2 * i + 1])
            selected = data.filter(isSelected);
            circle.classed("faded", d => !isSelected(d));
        }
        svg.property("value", selected).dispatch("input");
    }

    // If the brush is empty, select all circles.
    function brushended({selection}) {
        if (selection) return;
        svg.property("value", []).dispatch("input");
        circle.classed("faded", false);
    }
}

const svg = d3.create("svg")
    .attr("viewBox", [-padding, 0, width, width]);

svg.append("style")
    .text("circle.faded { fill: #000; fill-opacity: 1; r: 1px; }");

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


const defs = svg.append("svg:defs")

const marker = defs.selectAll("marker")
    .data(d3.range(1))
    .enter()
    .append("svg:marker")
    .attr("id", "arrow")
    .attr("markerHeight", 5)
    .attr("markerWidth", 5)
    .attr("markerUnits", "strokeWidth")
    .attr("orient", "auto")
    .attr("refX", 0)
    .attr("refY", 0)
    .attr("viewBox", "-5 -5 10 10")
    .append("svg:path")
    .attr("d", "M 0,0 m -5,-5 L 5,0 L -5,5 Z")
    .attr("fill", "grey");

const clip = defs.selectAll("clip")
    .data(d3.range(1))
    .enter()
    .append("svg:clipPath")
    .attr("id", "vector-field")
    .append("svg:rect")
    .attr("x", padding / 2 + 0.5)
    .attr("y", padding / 2 + 0.5)
    .attr("width", size - padding)
    .attr("height", size - padding);

cell.each(function (i) {
    // Vector field
    d3.select(this).selectAll("rect")
        .data(d3.range(10))
        .join("ellipse");

    d3.select(this).selectAll("ellipse")
        .attr("cx", size / 2 + 0.5)
        .attr("cy", size / 2 + 0.5)
        .attr("rx", j => (size - padding) * j / 10)
        .attr("ry", j => (size - padding) * j / 10)
        .attr("stroke-opacity", j => 0.8 - j / 8)
        .attr("stroke", "grey")
        .attr("clip-path", "url(#vector-field)");

    // Direction indicator
    d3.select(this).selectAll("rect")
        .data(d3.range(10))
        .join("svg:path");

    d3.select(this).selectAll("path")
        .attr("d", j => `M ${size / 2} ${size / 2 - (size - padding) * j / 10} z`)
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("stroke-linecap", "round")
        .attr("marker-start", "url(#arrow)")
        .attr("marker-end", "url(#arrow)")
        .attr("clip-path", "url(#vector-field)");

    d3.select(this).selectAll("circle")
        .data(data)
        .join("circle")
        .attr("cx", d => x[i](d.embedding[2 * i]))
        .attr("cy", d => y[i](d.embedding[2 * i + 1]));
});

const circle = cell.selectAll("circle")
    .attr("r", 3.5)
    .attr("fill-opacity", 0.7)
    .attr("fill", d => z(d.feature[colorBy]));

cell.call(brush, circle, svg);

svg.append("g")
    .style("font", "bold 10px sans-serif")
    .style("pointer-events", "none")
    .selectAll("text")
    .data(data.map(d => d.eigen).filter((_, index) => index % 2 == 0))
    .join("text")
    .attr("transform", (d, i) => `translate(${i * size + 20}, -27)`)
    .attr("x", padding)
    .attr("y", padding)
    .attr("dy", ".71em")
    .text((d, i) => `Disc ${i} (${d.toFixed(2)})`);

svg.property("value", [])

export const embedding_svg = svg;
