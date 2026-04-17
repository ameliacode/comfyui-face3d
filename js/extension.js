import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "MyNodes.extension",

    async setup() {
        // Runs once after ComfyUI is fully loaded.
        // Add event listeners, register sidebar tabs, commands, etc. here.
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Called for every node before it is registered.
        // Use to add custom widgets or modify node behavior per class.
        // Example:
        // if (nodeData.name === "ExampleImageNode") {
        //     const orig = nodeType.prototype.onNodeCreated;
        //     nodeType.prototype.onNodeCreated = function () {
        //         orig?.apply(this, arguments);
        //         this.addWidget("button", "Run", null, () => console.log("clicked"));
        //     };
        // }
    },

    nodeCreated(node) {
        // Called for each node instance after creation.
    },
});
