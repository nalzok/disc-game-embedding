"use strict";
import {embedding_svg} from "./embedding.js"
import {performance_svg} from "./performance.js"

const embedding_container = document.querySelector(".embedding-container")
const performance_container = document.querySelector(".performance-container")

embedding_container.appendChild(embedding_svg.node());
performance_container.appendChild(performance_svg.node());
