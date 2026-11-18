"""Record the README demo GIF from the real editor.

    pip install playwright pillow           # uses your installed Microsoft Edge
    build\src\Release\mg_server.exe --port 8711 --no-browser
    python tools/record_demo.py docs/assets/editor.gif [--url http://127.0.0.1:8711/]

Opens gpt2.json in a fresh browser profile, zooms into a transformer block,
expands the attention nest and changes d_model to 1024. A capture task takes
screenshots continuously while the actions run, and every frame keeps its
real duration. Use a separate server so the recording does not disturb a
session you have open.
"""
import argparse, asyncio, io, json, time
from playwright.async_api import async_playwright
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8711/")
args = ap.parse_args()
OUT, URL = args.out, args.url
W, H = 1440, 860
frames = []          # (t, png bytes)
stop = False

async def capture(pg):
    while not stop:
        frames.append((time.perf_counter(), await pg.screenshot(type="png")))

async def hold(s):
    await asyncio.sleep(s)

async def main():
    global stop
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge", headless=True)
        pg = await b.new_page(viewport={"width": W, "height": H}, color_scheme="dark")
        await pg.goto(URL)
        await pg.wait_for_selector(".react-flow__node", timeout=20000)
        await asyncio.sleep(1.0)
        await pg.click("text=Vertical"); await asyncio.sleep(1.2)
        await pg.click(".run-head"); await asyncio.sleep(0.4)
        await pg.keyboard.press("f"); await asyncio.sleep(0.8)

        cap = asyncio.create_task(capture(pg))
        await hold(1.8)                                   # the whole model
        att = pg.locator(".react-flow__node", has_text="attention").first
        box = await att.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] * 0.35
        await pg.mouse.move(x, y)
        for _ in range(9):                                # zoom into attention
            await pg.mouse.wheel(0, -120); await asyncio.sleep(0.12)
        await hold(1.5)
        await att.locator(".nest-toggle").click()       # open the nest in place
        await hold(1.2)
        q = pg.locator(".react-flow__node", has_text="query").first
        qb = await q.bounding_box()
        await pg.mouse.move(qb["x"] + qb["width"] / 2, qb["y"] + qb["height"] / 2)
        for _ in range(2):
            await pg.mouse.wheel(0, 120); await asyncio.sleep(0.12)
        await hold(2.0)
        await pg.locator(".react-flow__pane").click(position={"x": 30, "y": 30})   # deselect
        await hold(0.3)
        # change d_model in the inspector: shapes and params update live
        field = pg.locator("label.field.var", has_text="d_model").locator("input")
        await field.click(); await field.press("Control+a")
        await field.press_sequentially("1024", delay=110)
        await field.press("Enter")
        await hold(2.8)
        stop = True
        await cap
        await b.close()

    t0 = frames[0][0]
    imgs, durs = [], []
    for i, (t, png) in enumerate(frames):
        nxt = frames[i + 1][0] if i + 1 < len(frames) else t + 2.0
        im = Image.open(io.BytesIO(png)).convert("RGB").resize((1200, int(1200 * H / W)), Image.LANCZOS)
        imgs.append(im); durs.append(max(20, int((nxt - t) * 1000)))
    durs[-1] = 2500
    pal = imgs[len(imgs)//2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(OUT, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print(json.dumps({"frames": len(frames), "seconds": round(frames[-1][0]-t0, 2), "fps": round(len(frames)/(frames[-1][0]-t0),1)}))

asyncio.run(main())
