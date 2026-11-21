CSS = """
/******************** Main View CSS *********************/
div[data-testid="stStatusWidget"] div button {
    display: none;
} /* Hide redundant indicator */

.stAppHeader {
    display: none !important;
} /* Hide distracting header taking up space */

.stMainBlockContainer {
    height: 100% !important;
    padding-top: 0px !important;
    padding-inline: 20px !important;
} /* Fill the whole space */

iframe {
    height: 100vh !important;
} /* Fill the whole space */


/********************** Sidebar CSS **********************/    
[data-testid="stSidebarUserContent"] {
    padding-bottom: 0px !important;
}

[data-testid="stSidebarHeader"] {
    display: none !important;
} /* Hide collapse button */

:has(> * > .st-key-search-box) {
    display: flex !important;
    flex-direction: column !important;
    height: 100dvh !important;
    overflow-y: clip !important;
} /* Setup to restrict sidebar components to window height */

.stTextInput {
    padding-top: 10px !important;
} /* Search bar padding */

div[data-testid="InputInstructions"] > span {
    display: none;
} /* Hide "Press Enter to..." hint */

div:has(>.st-key-search-box) {
    flex-shrink: 0 !important;
} /* Lock search box to fixed height */

.st-key-search-box {
    gap: 0.25rem !important;
}

[data-testid="stCaptionContainer"]>p {f
    margin-bottom: 0.25rem !important;
}

[data-testid="stMarkdownContainer"] {
    text-align: left !important;
} /* Left justify result texts */

.stButton [data-testid="stMarkdownContainer"] {
    display: -webkit-box !important;
    -webkit-line-clamp: 2 !important;
    -webkit-box-orient: vertical !important;
    overflow: hidden !important;
} /* Limit results to 2 lines */

.st-key-search-results {
    gap: 0.5rem !important;
}

div:has(>.st-key-search-results) {
    flex-grow: 1 !important;
    overflow-y: auto !important;
} /* Fill available space and make the rest scrollable */

div:has(>.st-key-file-upload) {
    flex-shrink: 0 !important;
    padding-bottom: 15px !important;
} /* Lock file upload to fixed height */

[data-testid="stWidgetLabel"]:has(+[data-testid="stFileUploaderDropzone"]) {
    display: none !important;
} /* Hide unused label container */

[data-testid="stFileUploaderDropzone"] p {
    font-size: 0 !important;
} /* Remove English-only "Upload" text */

[data-testid="stFileUploaderDropzone"] p::after {
    content: 'Upload / Subir'  !important;
    font-size: 0.875rem !important;
} /* Add bilingual "Upload" label */

[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 0 !important;
} /* Remove English-only maximum file size disclaimer */

[data-testid="stFileUploaderDropzoneInstructions"]>div::after {
    content: 'Maximum / Máximo 200MB';
    font-size: 0.875rem;
    color: rgba(47, 45, 43, 0.7);
} /* Add bilingual file size disclaimer */

[data-testid="stFileUploaderDropzoneInput"]+div:not(:has([data-testid="stFileChips"])) {
    display: none !important;
} /* Remove hover hint text */
"""