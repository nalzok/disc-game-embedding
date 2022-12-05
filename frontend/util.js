'use strict';
import * as d3 from "https://cdn.skypack.dev/d3@7";
export {d3};

export const parm = {
  scatSize: 350,
};

export const glob = {
  csvData: {},
};

// dataGen Generates data
export const dataGen = function (dataRange, dataLen,
  // optional arguments
  { randomize = true, start = 'A', epsilon = 0.01 } = {}) {
  const offset = epsilon * (dataRange[1] - dataRange[0]);
  const scale = (1 - 2 * epsilon) * (dataRange[1] - dataRange[0]);
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/Array
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Spread_syntax
  return (
    [...Array(dataLen)]
      // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/fromCharCode
      // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/charCodeAt
      .map((_, i) => String.fromCharCode(start.charCodeAt(0) + i))
      .map((d, i) => ({
        cat: d,
        val: offset + scale * (randomize ? Math.random() : i / (dataLen - 1)),
      }))
  );
};

export const svgAppend = function (label, layout, Height) {
  const outer = d3
    .select('#allvis')
    .append('svg')
    .attr('id', `vis-${label}`) // modifies the svg
    .attr('width', layout.width + layout.margin.left + layout.margin.right)
    .attr(
      'height',
      (Height
        ? Height // want something other than layout.height
        : layout.height) +
      layout.margin.top +
      layout.margin.bottom
    );
  outer; // the newly-appended svg
    //.append('text')
    //.attr('class', 'label')
    //.attr('transform', 'translate(10,18)')
    //.text(label);
  const inner = outer // work within g that is translated by left,top margins
    .append('g')
    .attr('transform', `translate(${layout.margin.left}, ${layout.margin.top})`);
  if (!Height) {
    // not over-riding the height, so add a g in which to draw the Y axis scale
    inner.append('g').attr('class', 'axis axis--y').call(d3.axisLeft(layout.yScale));
  }
  inner // add a g in which to draw the X axis scale
    .append('g')
    .attr('class', 'axis axis--x')
    .attr('transform', `translate(0,${Height ? Height : layout.height})`)
    .call(d3.axisBottom(layout.xScale));
  return inner.append('g').attr('class', 'dots')
};
