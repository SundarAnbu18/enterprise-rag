/* Enterprise RAG chat widget loader.
 *
 * One static script serves every tenant:
 *
 *   <script src="https://YOUR-HOST/widget.js" data-tenant="SLUG" async></script>
 *
 * The tenant comes from the data-tenant attribute and the server origin from
 * the script's own src, so the snippet never goes stale — and there is no
 * API key in it: the iframe loads the hosted chat page, which authenticates
 * nothing and reveals nothing.
 *
 * Optional attributes: data-color="#1f6feb", data-position="left".
 */
(function () {
  "use strict";
  var script = document.currentScript;
  if (!script) { return; }
  var slug = script.getAttribute("data-tenant");
  if (!slug) {
    console.error("[erag-widget] missing data-tenant attribute on the script tag");
    return;
  }
  var origin;
  try { origin = new URL(script.src).origin; } catch (error) { return; }
  var color = script.getAttribute("data-color") || "#1f6feb";
  var side = script.getAttribute("data-position") === "left" ? "left" : "right";

  function mount() {
    var box = document.createElement("div");
    box.style.cssText =
      "position:fixed;bottom:24px;" + side + ":24px;z-index:2147483000;";

    var frame = document.createElement("iframe");
    frame.src = origin + "/chat/" + encodeURIComponent(slug) + "/";
    frame.title = "Chat assistant";
    frame.style.cssText =
      "display:none;width:380px;height:560px;max-width:calc(100vw - 48px);" +
      "max-height:calc(100vh - 110px);border:0;border-radius:16px;" +
      "box-shadow:0 12px 40px rgba(0,0,0,.25);margin-bottom:12px;background:#fff;";

    var button = document.createElement("button");
    button.type = "button";
    button.textContent = "💬";
    button.setAttribute("aria-label", "Open chat");
    button.style.cssText =
      "display:block;" + (side === "left" ? "margin-right:auto;" : "margin-left:auto;") +
      "width:56px;height:56px;border:0;border-radius:50%;cursor:pointer;" +
      "background:" + color + ";color:#fff;font-size:24px;" +
      "box-shadow:0 6px 20px rgba(0,0,0,.3);";

    button.addEventListener("click", function () {
      var open = frame.style.display !== "none";
      frame.style.display = open ? "none" : "block";
      button.textContent = open ? "💬" : "✕";
      button.setAttribute("aria-label", open ? "Open chat" : "Close chat");
    });

    box.appendChild(frame);
    box.appendChild(button);
    document.body.appendChild(box);
  }

  if (document.body) { mount(); }
  else { document.addEventListener("DOMContentLoaded", mount); }
})();
