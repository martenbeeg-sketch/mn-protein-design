import React from "react";
import { createRoot } from 'react-dom/client';
import StreamlitWrapper from "./StreamlitWrapper";
import MolstarCustomComponent from "./MolstarCustomComponent";
import { ContigSegment, StreamlitComponentValue, StructureVisualization } from "./types";

const container = document.getElementById('root');
const root = createRoot(container!);

if (container?.classList?.contains("molstar_notebook")) {
    const structures = (window as any).STRUCTURES as StructureVisualization[];
    const allContigsParsed: ContigSegment[][] = [];
    structures.forEach((s: any) => allContigsParsed.push(s["contigs"] ?? []));

    const updateStreamlitComponentValue = (val: StreamlitComponentValue) => { };

    root.render(
        <div id="molstar_root">
            <MolstarCustomComponent
                structures={structures}
                divName="molstar_root"
                contigs={allContigsParsed}
                highlightedContig={null}
                selectionMode={false}
                showControls={false}
                updateStreamlitComponentValue={updateStreamlitComponentValue}
            />
        </div>
    );
} else {
    root.render(
        <React.StrictMode>
            <StreamlitWrapper />
        </React.StrictMode>
    );
}
