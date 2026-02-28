# EfficientManim — MCP & AI Features Guide

A plain-English guide to using the AI Panel, Auto Voiceover, and MCP Agent Mode from inside the app. No coding knowledge required for the first half. The second half covers the MCP Agent Mode for power users.

---

## Table of Contents

1. [What Changed in This Update](#1-what-changed-in-this-update)
2. [The AI Panel — Where Everything Lives](#2-the-ai-panel--where-everything-lives)
3. [Normal AI Mode — Generating Manim Code](#3-normal-ai-mode--generating-manim-code)
4. [The Voiceover Tab — Adding Audio to Your Animations](#4-the-voiceover-tab--adding-audio-to-your-animations)
   - [Generating Audio](#generating-audio)
   - [Previewing Audio](#previewing-audio)
   - [Attaching Audio to a Node](#attaching-audio-to-a-node)
5. [Auto Voiceover — One Click, Entire Project](#5-auto-voiceover--one-click-entire-project)
6. [MCP Agent Mode — AI That Edits Your Scene Directly](#6-mcp-agent-mode--ai-that-edits-your-scene-directly)
   - [What It Does](#what-it-does)
   - [How to Use It Step by Step](#how-to-use-it-step-by-step)
   - [What Kinds of Instructions Work Well](#what-kinds-of-instructions-work-well)
   - [Reading the Results Panel](#reading-the-results-panel)
7. [Tips & Troubleshooting](#7-tips--troubleshooting)

---

## 1. What Changed in This Update

Here is a quick summary of everything that was added or changed so you know what to look for when you open the app.

**AI Panel moved to the left side.** It used to be a tab in the right panel. It is now a permanent panel docked on the left side of the window. It cannot be dragged, undocked, or closed — it is always there.

**Voiceover works on every node type.** Before, you could only attach audio to Text and Tex nodes. Now every node in your scene — Circles, FadeIns, Transforms, everything — can have a voiceover attached to it.

**The Voiceover tab got a full audio player.** After generating audio you can now preview it directly inside the app with a play button, pause button, stop button, a seek slider, and a timer showing current position and total duration.

**"Add to Animation Node" button.** There is a new green button in the Voiceover tab that attaches your generated audio to whichever node you have selected in the dropdown. Before you had to generate and it would try to auto-attach. Now it is two separate deliberate steps.

**Auto Voiceover checkbox in the AI Panel.** Tick one checkbox and Gemini will go through every single node in your scene, write a voiceover script for each one, generate the audio, and attach it — completely automatically. Your whole project ends up render-ready with synced audio in one go.

**MCP Agent Mode checkbox in the AI Panel.** A new mode where instead of generating Manim code, Gemini actually operates the application directly — creating nodes, changing parameters, and building your scene for you.

---

## 2. The AI Panel — Where Everything Lives

When you open the app, look at the **left side of the window**. The AI Panel is the tall panel docked there permanently. It has a blue header that says **🤖 AI Assistant**.

Inside the AI Panel you will see:

- A text box where you type your prompt
- Two checkboxes just below the text box — **Enable Auto Voiceover** and **MCP Agent Mode**
- The **⚡ Generate Code** button
- A scrollable response area below the button

Everything AI-related flows through this one panel. You do not need to switch tabs to access it.

---

## 3. Normal AI Mode — Generating Manim Code

This is the default behavior — nothing special to enable.

**Step 1.** Make sure both checkboxes (**Auto Voiceover** and **MCP Agent Mode**) are **unchecked**.

**Step 2.** Click inside the prompt text box and type what you want. Be specific. Example:

> *Create a red square that rotates 90 degrees, then a blue circle that fades in next to it.*

**Step 3.** Click **⚡ Generate Code**.

**Step 4.** Watch the response area fill in. Gemini streams the code back to you in real time.

**Step 5.** Once it finishes, the **✅ Merge to Scene** button lights up. Click it to add the generated nodes to your scene graph.

**Step 6.** If you do not like the result, click **❌ Reject** and try a different prompt.

> **Tip:** The more specific your prompt, the better the result. Instead of "make an animation," try "make a MathTex equation that writes itself on screen over 2 seconds, then fades out."

---

## 4. The Voiceover Tab — Adding Audio to Your Animations

Click the **🎙️ Voiceover** tab in the right panel. This is the AI TTS (text-to-speech) studio.

### Generating Audio

**Step 1.** Choose a voice from the **Voice** dropdown at the top. The options are Puck, Charon, Kore, Fenrir, Aoede, and Zephyr. Zephyr is selected by default — it is a clear, neutral voice that works well for most animations.

**Step 2.** Type your script in the **Script** text box. Write it like you are talking out loud, not like code. Example:

> *A red square appears and rotates ninety degrees clockwise.*

**Step 3.** Click **⚡ Generate Audio**. A thin progress bar appears under the button while Gemini generates the audio. This usually takes 3–8 seconds depending on the length of your script.

**Step 4.** When it finishes, the progress bar disappears and the audio player in the middle of the panel becomes active.

---

### Previewing Audio

Once audio is generated, the player in the **🎵 Audio Preview** section becomes usable.

- **▶ Play** — starts playback
- **⏸ Pause** — pauses (click ▶ again to resume)
- **⏹ Stop** — stops and resets to the beginning
- **The slider** — drag it to jump to any position in the audio
- **The time display** — shows your current position and total duration in MM:SS format, for example `00:04 / 00:09`

Listen to it. If the voice sounds off or the pacing is wrong, edit your script and click Generate Audio again to re-generate.

---

### Attaching Audio to a Node

Once you are happy with the audio, attach it to a node in your scene.

**Step 1.** Look at the **🔗 Attach to Animation Node** section below the player.

**Step 2.** Open the dropdown. It lists every node currently in your scene. Animation nodes show a 🎬 icon, Mobject nodes show a 🔷 icon. Every type is available — not just text nodes.

**Step 3.** Select the node this voiceover belongs to.

**Step 4.** Click the green **📎 Add to Animation Node** button.

A confirmation dialog appears telling you the audio was attached and showing the detected duration. Click OK.

The node now has the audio file linked to it. When you render, the audio duration is used for timing. You can see it in the Properties panel if you select the node.

---

## 5. Auto Voiceover — One Click, Entire Project

This feature uses Gemini as an autonomous agent that handles the entire voiceover workflow for every node in your project at once.

**When to use it:** You have finished building your scene and you want to add narration to all of it without writing scripts or attaching audio manually.

**Step 1.** Make sure your scene is complete and your nodes are named in a way that makes sense. Auto Voiceover uses the node names and class types to write the scripts — a node called "intro_circle" gets a better script than one called "node_4."

**Step 2.** Go to the AI Panel on the left side.

**Step 3.** Tick the checkbox that says **🎙️ Enable Auto Voiceover**.

**Step 4.** In the prompt box, you can optionally type a style note like *"keep scripts under 8 words each"* or *"use an enthusiastic tone."* You can also leave it blank.

**Step 5.** Click **⚡ Generate Code** (the button label does not change when modes are active).

**What happens next:**

The response area shows you what the agent is doing in real time. It goes through these stages automatically:

1. Reads all nodes in your scene
2. Sends them to Gemini, which writes a short voiceover script for every node
3. The scripts appear in the response panel so you can read them
4. For each script, audio is generated one at a time via the TTS system
5. Each audio file is attached to its corresponding node
6. When all nodes are done, the graph compiles automatically

At the end you see a message that says **🎬 Auto Voiceover complete! All nodes have been synced and the project is render-ready.** A dialog also pops up confirming this.

**Your project is now fully voiceover-synced.** Every node has an audio file attached. Durations are stored. You can hit render.

> **Note:** Auto Voiceover generates audio sequentially, one node at a time. If you have 10 nodes it might take a minute or two. You can watch the progress in the response panel — each node shows a green ✅ when its audio is attached.

---

## 6. MCP Agent Mode — AI That Edits Your Scene Directly

### What It Does

In normal mode, Gemini generates Python code and you merge it into the scene. In MCP Agent Mode, there is no code generation step. Instead, Gemini looks at your current scene, figures out what needs to happen, and then **directly creates nodes, sets parameters, and modifies your scene** as if it had a mouse and keyboard.

Think of it like this: normal mode hands you blueprints. MCP Agent Mode sends in a contractor who builds it for you.

---

### How to Use It Step by Step

**Step 1.** Open the AI Panel on the left side of the window.

**Step 2.** Tick the checkbox that says **🔌 MCP Agent Mode**.

**Step 3.** Type your instruction in the prompt box. Write it naturally, like you are asking a colleague. Examples:

> *Add a large blue circle in the center and a FadeIn animation for it.*

> *Change the color of the node called "title_text" to yellow.*

> *Add a MathTex node showing the Pythagorean theorem and a Write animation.*

**Step 4.** Click **⚡ Generate Code**.

**Step 5.** The response panel shows Gemini's raw command plan streaming in. It looks like a JSON list — that is normal, that is the list of actions Gemini decided to take.

**Step 6.** Once streaming finishes, the app automatically executes every command. The response panel updates to show the results:

- A **green ✅** next to each command that succeeded
- A **red ❌** next to any command that failed, with a short reason

**Step 7.** Look at your scene graph. The nodes Gemini created are already there. The graph compiles automatically at the end.

---

### What Kinds of Instructions Work Well

**Good — specific and actionable:**
- *"Add a red square named box1 and a FadeIn animation for it"*
- *"Create a Text node that says 'Hello World' and a Write animation"*
- *"Add a Transform animation between the two circles already in the scene"*
- *"Switch to the Properties tab"*
- *"Save the project"*

**Less good — too vague:**
- *"Make it look better"*
- *"Do something cool"*
- *"Fix the animation"*

**Good for modifications (using existing node names):**
- *"Set the color of the node called 'hero_circle' to gold"*
- *"Rename the node called 'node_1' to 'intro_shape'"*

> **Tip:** Name your nodes clearly as you build. Gemini can see all the node names in your scene and uses them when planning modifications. A scene with nodes named "intro_circle," "title_text," and "fade_in_title" is much easier for it to work with than one full of "node_1," "node_2," "node_3."

---

### Reading the Results Panel

After execution the response panel shows a summary that looks like this:

```
🔌 MCP Agent Mode
Instruction: Add a blue circle and fade it in

[...Gemini's command plan streams here...]

Executing 3 commands:
✅ create_node → {'id': 'f3d875...', 'name': 'blue_circle'}
✅ create_node → {'id': '9a2bc3...', 'name': 'fade_in_1'}
✅ compile_graph → {'message': 'Graph compiled.'}

Agent finished. 3/3 commands executed successfully.
```

If something goes wrong it looks like:

```
✅ create_node → {'id': 'f3d875...', 'name': 'blue_circle'}
❌ set_node_param — Node not found: '???'
✅ compile_graph

Agent finished. 2/3 commands executed successfully.
```

The error message tells you what went wrong. In the example above, Gemini tried to reference a node by an ID it guessed incorrectly. You can simply try the instruction again — Gemini re-reads the scene each time including the newly created node from the first attempt, so it usually succeeds on the second try.

---

## 7. Tips & Troubleshooting

---

**The AI Panel is gone / I can't find it.**
It is on the left side of the main window as a permanent docked panel. It cannot be closed or moved. If your window is very narrow it might be squished — try making the window wider or dragging the splitter between the left panel and the canvas.

---

**Auto Voiceover finished but some nodes show ❌.**
This usually means the TTS API had a hiccup on one node. The other nodes are fine. You can manually go to the Voiceover tab, type a script for the missed node, generate audio, and attach it with the **📎 Add to Animation Node** button.

---

**The audio preview player does not have any sound.**
Make sure your system volume is up and no other app is holding the audio device exclusively. The player uses your default system audio output. Try clicking Stop and then Play again.

---

**MCP Agent Mode created the wrong nodes.**
Try being more specific in your prompt. Mention the exact Manim class names if you know them — for example "a Circle mobject" instead of just "a circle." Also check that your existing nodes have meaningful names, since Gemini reads them to understand the scene context.

---

**"Generate Code" does nothing when I click it.**
Check that you have typed something in the prompt box. An empty prompt is ignored silently. Also check your Gemini API key is set — go to **File → Settings** and make sure the API key field is filled in.

---

**The voiceover audio duration shows 0.0 seconds.**
This happens if the `pydub` library is not installed. The audio file still gets attached correctly — only the duration metadata is missing. To fix it, run `pip install pydub` in your terminal and restart the app.

---

**I attached audio to the wrong node by accident.**
Go to the Voiceover tab, select the correct node in the dropdown, and click **📎 Add to Animation Node** again with the right audio loaded. It will overwrite the previous attachment. Alternatively, use MCP Agent Mode with the instruction *"remove the voiceover from the node called [name]"*.

---

**MCP Agent Mode says "Cannot connect to main window."**
This is rare and usually means the app was in a loading state when you clicked Generate. Wait a moment and try again.

---

**Auto Voiceover is taking a very long time.**
TTS generation is done one node at a time to avoid hitting API rate limits. A scene with 15 nodes might take 3–4 minutes. This is normal. You can watch the progress in the response panel — each completed node shows ✅. The app remains fully usable while it runs in the background.
