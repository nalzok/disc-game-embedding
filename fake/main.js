"use strict";
import * as d3 from "https://cdn.skypack.dev/d3@7";
import {data} from "./data.js";

// config
const colorBy = 0
const firstN = 4

const padding = 28
const width = 954
const columns = d3.range(firstN)
const size = (width - (columns.length + 1) * padding) / columns.length + padding

const x = columns.map(c => d3.scaleLinear()
    .domain(d3.extent(data, d => d.embedding[2 * c]))
    .rangeRound([padding / 2, size - padding / 2]))

const y = columns.map(c => d3.scaleLinear()
    .domain(d3.extent(data, d => d.embedding[2 * c + 1]))
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
            circle.classed("hidden", d => !isSelected(d));
        }
        svg.property("value", selected).dispatch("input");
    }

    // If the brush is empty, select all circles.
    function brushended({selection}) {
        if (selection) return;
        svg.property("value", []).dispatch("input");
        circle.classed("hidden", false);
    }
}

const svg = d3.create("svg")
    .attr("viewBox", [-padding, 0, width, width]);

svg.append("style")
    .text(`circle.hidden { fill: #000; fill-opacity: 1; r: 1px; }`);

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


cell.each(function (i) {
    d3.select(this).selectAll("circle")
        .data(d3.range(1, 5))
        .join("ellipse")
        .attr("cx", size / 2 + 0.5)
        .attr("cy", size / 2 + 0.5)
        .attr("rx", i => (size - padding) * (i + 1) / 10)
        .attr("ry", i => (size - padding) * (i + 1) / 10)
        .attr("stroke-opacity", 0.2)
        .attr("stroke", "black");

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
    .data(columns)
    .join("text")
    .attr("transform", (d, i) => `translate(${i * size},0)`)
    .attr("x", padding)
    .attr("y", padding)
    .attr("dy", ".71em")
    .text(d => d);

svg.property("value", [])

const container = document.querySelector(".container")
container.appendChild(svg.node());