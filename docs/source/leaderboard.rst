Leaderboard 🥇
===========

The leaderboard page is automatically refreshed by the docs workflow whenever files in ``csv_results/`` change.

CSV Leaderboard Viewer
----------------------

.. raw:: html

   <div id="leaderboard-viewer" style="margin: 1rem 0 2rem 0;">
     <label for="csv-file-select"><strong>CSV file:</strong></label>
     <select id="csv-file-select" style="margin: 0 0.75rem 0 0.5rem;"></select>

     <div style="margin-top: 0.75rem;">
       <label for="csv-structured-filter"><strong>Structured filter:</strong></label>
       <input
         id="csv-structured-filter"
         type="text"
         placeholder="method={CellMNN, scNODE},dataset={SuoDataset, MaDataset}"
         style="margin-left: 0.5rem; min-width: 520px; max-width: 100%;"
       />
       <p style="margin: 0.4rem 0 0 0; font-size: 0.92rem; color: #444;">
         Example: <code>method={CellMNN, scNODE},dataset={SuoDataset}</code>
         means (method is CellMNN OR scNODE) AND (dataset is SuoDataset).
         Column names are case-insensitive.
       </p>
     </div>

     <p style="margin: 0.5rem 0 0.25rem 0;"><strong>Download all CSVs:</strong> <span id="csv-download-all"></span></p>

     <p id="csv-status" style="margin-top: 0.75rem;"></p>

     <div style="overflow-x: auto; max-width: 100%; border: 1px solid #ddd; border-radius: 8px;">
       <table id="csv-table" style="border-collapse: collapse; width: 100%; min-width: 900px;"></table>
     </div>
   </div>

   <script>
   (function () {
     const csvFiles = [
       "embedding.csv",
       "gex_pred.csv",
       "graph_sim.csv",
     ];

     const selectEl = document.getElementById("csv-file-select");
     const downloadAllEl = document.getElementById("csv-download-all");
     const structuredFilterEl = document.getElementById("csv-structured-filter");
     const statusEl = document.getElementById("csv-status");
     const tableEl = document.getElementById("csv-table");
    const PAGE_SIZE = 50;

     let fullRows = [];
     let headers = [];
     let sortState = { index: -1, ascending: true };

     function parseCsvLine(line) {
       const out = [];
       let current = "";
       let inQuotes = false;

       for (let i = 0; i < line.length; i++) {
         const ch = line[i];
         const next = line[i + 1];

         if (ch === '"') {
           if (inQuotes && next === '"') {
             current += '"';
             i++;
           } else {
             inQuotes = !inQuotes;
           }
         } else if (ch === "," && !inQuotes) {
           out.push(current);
           current = "";
         } else {
           current += ch;
         }
       }

       out.push(current);
       return out;
     }

     function parseCsv(text) {
       const lines = text
         .split(/\r?\n/)
         .map((l) => l.trimEnd())
         .filter((l) => l.length > 0);

       if (!lines.length) {
         return { headers: [], rows: [] };
       }

       const parsedHeaders = parseCsvLine(lines[0]);
       const rows = lines.slice(1).map(parseCsvLine);
       return { headers: parsedHeaders, rows: rows };
     }

     function splitOutsideBraces(input) {
       const parts = [];
       let current = "";
       let braceDepth = 0;

       for (let i = 0; i < input.length; i++) {
         const ch = input[i];
         if (ch === "{") {
           braceDepth += 1;
           current += ch;
         } else if (ch === "}") {
           braceDepth = Math.max(0, braceDepth - 1);
           current += ch;
         } else if (ch === "," && braceDepth === 0) {
           parts.push(current.trim());
           current = "";
         } else {
           current += ch;
         }
       }

       if (current.trim()) {
         parts.push(current.trim());
       }

       return parts;
     }

     function parseStructuredFilters(input) {
       const text = input.trim();
       if (!text) {
         return { filters: [], error: null };
       }

       const clauses = splitOutsideBraces(text);
       const filters = [];

       for (const clause of clauses) {
         const eqIdx = clause.indexOf("=");
         if (eqIdx <= 0) {
           return { filters: [], error: `Invalid filter clause: ${clause}` };
         }

         const key = clause.slice(0, eqIdx).trim().toLowerCase();
         const valueExpr = clause.slice(eqIdx + 1).trim();
         if (!key) {
           return { filters: [], error: `Missing column name in clause: ${clause}` };
         }

         let values = [];
         if (valueExpr.startsWith("{") && valueExpr.endsWith("}")) {
           const inner = valueExpr.slice(1, -1);
           values = inner
             .split(",")
             .map((v) => v.trim())
             .filter((v) => v.length > 0);
         } else {
           values = [valueExpr];
         }

         if (!values.length) {
           return { filters: [], error: `No values provided for '${key}'` };
         }

         filters.push({ key: key, values: values.map((v) => v.toLowerCase()) });
       }

       return { filters: filters, error: null };
     }

     function compareValues(a, b) {
       const aNum = Number(a);
       const bNum = Number(b);
       const bothNumeric = Number.isFinite(aNum) && Number.isFinite(bNum);

       if (bothNumeric) {
         return aNum - bNum;
       }

       return String(a).localeCompare(String(b));
     }

     function renderTable(allDisplayRows) {
       const displayRows = allDisplayRows.slice(0, PAGE_SIZE);
       tableEl.innerHTML = "";

       if (!headers.length) {
         statusEl.textContent = "No data available.";
         return;
       }

       const thead = document.createElement("thead");
       const headerRow = document.createElement("tr");

       headers.forEach((header, index) => {
         const th = document.createElement("th");
         const sortedHere = sortState.index === index;
         const arrow = sortedHere ? (sortState.ascending ? " ▲" : " ▼") : "";
         th.textContent = header + arrow;
         th.style.borderBottom = "1px solid #ddd";
         th.style.padding = "0.5rem";
         th.style.textAlign = "left";
         th.style.cursor = "pointer";
         th.style.whiteSpace = "nowrap";

         th.addEventListener("click", () => {
           if (sortState.index === index) {
             sortState.ascending = !sortState.ascending;
           } else {
             sortState.index = index;
             sortState.ascending = true;
           }
           applyFiltersAndSort();
         });

         headerRow.appendChild(th);
       });

       thead.appendChild(headerRow);
       tableEl.appendChild(thead);

       const tbody = document.createElement("tbody");
       displayRows.forEach((row) => {
         const tr = document.createElement("tr");
         row.forEach((cell) => {
           const td = document.createElement("td");
           td.textContent = cell;
           td.style.padding = "0.45rem 0.5rem";
           td.style.borderBottom = "1px solid #f0f0f0";
           td.style.whiteSpace = "nowrap";
           tr.appendChild(td);
         });
         tbody.appendChild(tr);
       });

       tableEl.appendChild(tbody);
       if (allDisplayRows.length > PAGE_SIZE) {
         statusEl.textContent = `Showing top ${displayRows.length} of ${allDisplayRows.length} filtered rows (${fullRows.length} total).`;
       } else {
         statusEl.textContent = `Showing ${displayRows.length} of ${fullRows.length} total rows.`;
       }
     }

     function applyFiltersAndSort() {
       const structuredQuery = structuredFilterEl.value;
       const parsedFilters = parseStructuredFilters(structuredQuery);
       let rows = fullRows.slice();

       if (parsedFilters.error) {
         tableEl.innerHTML = "";
         statusEl.textContent = parsedFilters.error;
         return;
       }

       if (parsedFilters.filters.length) {
         const headerIndexByName = {};
         headers.forEach((h, i) => {
           headerIndexByName[String(h).toLowerCase()] = i;
         });

         const unknownColumns = parsedFilters.filters
           .map((f) => f.key)
           .filter((key) => !(key in headerIndexByName));

         if (unknownColumns.length) {
           tableEl.innerHTML = "";
           statusEl.textContent = `Unknown column(s): ${unknownColumns.join(", ")}`;
           return;
         }

         rows = rows.filter((row) =>
           parsedFilters.filters.every((f) => {
             const idx = headerIndexByName[f.key];
             const cell = String(row[idx] ?? "").trim().toLowerCase();
             return f.values.includes(cell);
           })
         );
       }

       if (sortState.index >= 0) {
         rows.sort((rowA, rowB) => {
           const left = rowA[sortState.index] ?? "";
           const right = rowB[sortState.index] ?? "";
           const cmp = compareValues(left, right);
           return sortState.ascending ? cmp : -cmp;
         });
       }

       renderTable(rows);
     }

     function csvUrl(fileName) {
       return `_static/leaderboard/csv/${fileName}`;
     }

     function renderAllDownloadLinks() {
       downloadAllEl.innerHTML = "";
       csvFiles.forEach((fileName, idx) => {
         const link = document.createElement("a");
         link.href = csvUrl(fileName);
         link.textContent = fileName;
         link.setAttribute("download", fileName);
         downloadAllEl.appendChild(link);

         if (idx < csvFiles.length - 1) {
           downloadAllEl.appendChild(document.createTextNode(" | "));
         }
       });
     }

     async function loadCsv(fileName) {
       const url = csvUrl(fileName);
       statusEl.textContent = `Loading ${fileName}...`;
       tableEl.innerHTML = "";

       try {
         const response = await fetch(url, { cache: "no-cache" });
         if (!response.ok) {
           throw new Error(`HTTP ${response.status}`);
         }

         const text = await response.text();
         const parsed = parseCsv(text);
         headers = parsed.headers;
         fullRows = parsed.rows;
         sortState = { index: -1, ascending: true };
         applyFiltersAndSort();
       } catch (err) {
         headers = [];
         fullRows = [];
         tableEl.innerHTML = "";
         statusEl.textContent = `Unable to load ${fileName}. ${err.message}`;
       }
     }

     csvFiles.forEach((fileName) => {
       const option = document.createElement("option");
       option.value = fileName;
       option.textContent = fileName;
       selectEl.appendChild(option);
     });

     renderAllDownloadLinks();

     selectEl.addEventListener("change", () => loadCsv(selectEl.value));
     structuredFilterEl.addEventListener("input", applyFiltersAndSort);

     if (csvFiles.length) {
       selectEl.value = csvFiles[0];
       loadCsv(csvFiles[0]);
     } else {
       statusEl.textContent = "No CSV files configured for viewer.";
     }
   })();
   </script>

Leaderboard Plots
-----------------

The plots below are generated from ``csv_results/*.csv`` using ``scTimeBench --plot_from_csv`` during docs automation.

.. raw:: html

   <div id="leaderboard-plots" style="display: grid; grid-template-columns: 1fr; gap: 1rem;"></div>

   <script>
   (function () {
     const plotFiles = [
       "log_fold_change_heatmap.svg",
       "graph_sim_scatter.svg",
     ];

     const container = document.getElementById("leaderboard-plots");

     async function addIfExists(fileName) {
       const src = `_static/leaderboard/plots/${fileName}`;
       try {
         const response = await fetch(src, { method: "HEAD", cache: "no-cache" });
         if (!response.ok) {
           return;
         }

         const wrapper = document.createElement("figure");
         wrapper.style.margin = "0";

         const title = document.createElement("figcaption");
         title.textContent = fileName;
         title.style.fontWeight = "600";
         title.style.marginBottom = "0.5rem";

         const image = document.createElement("img");
         image.src = src;
         image.alt = fileName;
         image.style.width = "100%";
         image.style.border = "1px solid #ddd";
         image.style.borderRadius = "8px";

         wrapper.appendChild(title);
         wrapper.appendChild(image);
         container.appendChild(wrapper);
       } catch (e) {
         // Ignore missing files and network errors.
       }
     }

     Promise.all(plotFiles.map(addIfExists)).then(() => {
       if (!container.children.length) {
         const p = document.createElement("p");
         p.textContent = "No leaderboard plots were found in static assets yet.";
         container.appendChild(p);
       }
     });
   })();
   </script>
