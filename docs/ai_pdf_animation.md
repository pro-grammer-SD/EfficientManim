# AI PDF Animation

### Overview
Attach one or multiple PDFs in the AI Assistant tab and automatically generate animated Manim scenes. The feature converts static slide content into a structured animation plan, builds runnable Manim code, and loads the result into the node editor for further editing.

### How It Works
1. Attach PDFs in the AI Assistant tab.
2. Extract slide structure from each page.
3. Send structured content plus the user prompt to the AI model.
4. Generate slide-level animation instructions in JSON.
5. Convert the instructions into Manim scenes.
6. Convert the scenes into node graphs and load them into the editor.

### Supported Input
- Lecture slides
- Math PDFs
- Educational PDFs
- Slide exports from PowerPoint or Google Slides

### Example Workflow
A student uploads a calculus lecture PDF, adds the prompt “animate the definition of the derivative and show the limit process,” and clicks Generate. The app parses each page, produces a slide-by-slide animation plan, creates Manim scenes, and opens the resulting node graph for refinement.

### Technical Details
PDF parsing uses `pdfplumber` to extract page text and a lightweight classifier to identify headings, bullets, and equations in order. The AI prompt is built by `AIContextBuilder`, which compresses large pages and merges all PDFs with the user prompt. `AISlideAnimator` calls the AI API using `GEMINI_API_KEY` and returns structured JSON. `SlidesToManim` turns that JSON into runnable Manim code, and `SlidesToNodes` creates node groups, animations, and wiring per slide inside the editor.
