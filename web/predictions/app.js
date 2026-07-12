"use strict";

const API_URL = "/api/predictions/upcoming";
const CARD_SECTION_ORDER = ["main_event", "main_card", "prelims", "early_prelims", "fight_card"];
const CARD_SECTION_LABELS = {
  main_event: "Main Event",
  main_card: "Main Card",
  prelims: "Prelims",
  early_prelims: "Early Prelims",
  fight_card: "Fight Card"
};

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = text;
  return element;
}

function formatPercentage(probability) {
  return `${(probability * 100).toFixed(1)}%`;
}

function formatDate(dateValue) {
  const parsed = new Date(`${dateValue}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return dateValue;
  return new Intl.DateTimeFormat(undefined, {
    weekday: "long", year: "numeric", month: "long", day: "numeric", timeZone: "UTC"
  }).format(parsed);
}

function appendMetadata(list, label, value) {
  if (!value) return;
  list.append(createElement("dt", "", label), createElement("dd", "", value));
}

function buildFighter(fight, side) {
  const isA = side === "a";
  const name = isA ? fight.fighter_a : fight.fighter_b;
  const probability = isA ? fight.fighter_a_probability : fight.fighter_b_probability;
  const isWinner = fight.predicted_winner_side === side;
  const fighter = createElement("div", `fighter fighter-${side}${isWinner ? " winner" : ""}`);
  fighter.dataset.testid = `fighter-${side}`;
  fighter.append(
    createElement("p", "fighter-name", name),
    createElement("p", "fighter-probability", formatPercentage(probability)),
    createElement("span", "winner-marker", isWinner ? "Model pick" : "")
  );
  return fighter;
}

function buildFightCard(fight) {
  if (!fight.prediction_available) {
    const card = createElement("article", "fight-card unavailable");
    card.dataset.testid = "fight-card";
    const top = createElement("div", "card-topline");
    top.append(
      createElement("span", "bout-label", fight.bout_label || "UFC Bout"),
      createElement("span", "unavailable-badge", "Prediction unavailable")
    );
    const names = createElement("div", "unavailable-fighters");
    names.append(
      createElement("p", "fighter-name", fight.fighter_a),
      createElement("span", "versus", "vs."),
      createElement("p", "fighter-name", fight.fighter_b)
    );
    card.append(
      top,
      names,
      createElement("p", "unavailable-reason", fight.prediction_unavailable_reason || "No validated prediction is available.")
    );
    return card;
  }
  const card = createElement("article", "fight-card");
  card.dataset.testid = "fight-card";

  const top = createElement("div", "card-topline");
  top.append(
    createElement("span", "bout-label", fight.bout_label || "UFC Bout"),
    createElement("span", "confidence", fight.confidence_label)
  );

  const fighters = createElement("div", "fighters");
  fighters.append(buildFighter(fight, "a"), buildFighter(fight, "b"));

  const bar = createElement("div", "probability-bar");
  bar.dataset.testid = "probability-bar";
  bar.setAttribute(
    "aria-label",
    `${fight.fighter_a} ${formatPercentage(fight.fighter_a_probability)}, ${fight.fighter_b} ${formatPercentage(fight.fighter_b_probability)}`
  );
  const barA = createElement("span", "bar-a");
  const barB = createElement("span", "bar-b");
  barA.style.width = `${fight.fighter_a_probability * 100}%`;
  barB.style.width = `${fight.fighter_b_probability * 100}%`;
  bar.append(barA, barB);

  const pick = createElement("div", "pick-line");
  pick.append(
    createElement("span", "pick-label", "Model pick"),
    createElement("span", "pick-name", fight.predicted_winner || "Even matchup")
  );

  const footer = createElement("div", "card-footer");
  footer.append(createElement("span", "frozen-badge", "Official frozen prediction"));
  if (fight.prediction_created_at) {
    const time = createElement("time", "pick-label", "Recorded before the event");
    time.dateTime = fight.prediction_created_at;
    footer.append(time);
  }

  card.append(top, fighters, bar, pick, footer);
  return card;
}

function buildEventSection(event) {
  const section = createElement("section", "event-section");
  section.dataset.testid = "event-section";
  section.setAttribute("aria-labelledby", `${event.event_id}-title`);

  const header = createElement("header", "event-header");
  const heading = createElement("div");
  heading.append(
    createElement("p", "event-date", formatDate(event.event_date)),
    createElement("h2", "event-title", event.event_name)
  );
  heading.lastElementChild.id = `${event.event_id}-title`;
  header.append(
    heading,
    createElement("span", "event-count", `${event.predicted_fight_count} predictions / ${event.fight_count} fights`)
  );

  const cardSections = createElement("div", "card-sections");
  const groupedFights = new Map();
  event.fights.forEach((fight) => {
    const key = CARD_SECTION_LABELS[fight.card_section] ? fight.card_section : "fight_card";
    if (!groupedFights.has(key)) groupedFights.set(key, []);
    groupedFights.get(key).push(fight);
  });
  CARD_SECTION_ORDER.forEach((key) => {
    const sectionFights = groupedFights.get(key);
    if (!sectionFights?.length) return;
    const cardSection = createElement("section", "card-section");
    cardSection.dataset.cardSection = key;
    cardSection.append(createElement("h3", "card-section-heading", CARD_SECTION_LABELS[key]));
    const fights = createElement("div", "fight-list");
    sectionFights.forEach((fight) => fights.append(buildFightCard(fight)));
    cardSection.append(fights);
    cardSections.append(cardSection);
  });

  const details = createElement("details", "event-details");
  details.append(createElement("summary", "", "Model details"));
  const metadata = createElement("dl", "metadata-grid");
  appendMetadata(metadata, event.batch_ids?.length > 1 ? "Batches" : "Batch", (event.batch_ids || [event.batch_id]).join(", "));
  appendMetadata(metadata, "Model", (event.model_versions || [event.model_version]).join(", "));
  appendMetadata(metadata, "Calibration", (event.calibration_versions || [event.calibration_version]).join(", "));
  appendMetadata(metadata, "Prediction status", "Official frozen prediction");
  appendMetadata(metadata, "Recorded", event.prediction_created_at);
  details.append(metadata);

  section.append(header, cardSections, details);
  return section;
}

function showStatus(message, isError = false) {
  const status = document.getElementById("dashboard-status");
  status.className = `status-panel${isError ? " error" : ""}`;
  status.replaceChildren(createElement("span", "", message));
  status.hidden = false;
}

function renderDashboard(payload) {
  const status = document.getElementById("dashboard-status");
  const eventsRoot = document.getElementById("events");
  const warning = document.getElementById("dashboard-warning");
  eventsRoot.replaceChildren();
  warning.hidden = true;

  if (!payload.events || payload.events.length === 0) {
    showStatus("No upcoming official predictions are currently available.");
    return;
  }

  status.hidden = true;
  payload.events.forEach((event) => eventsRoot.append(buildEventSection(event)));
  const invalidCount = payload.diagnostics?.invalid_row_count || 0;
  if (invalidCount > 0) {
    warning.textContent = `${invalidCount} malformed prediction ${invalidCount === 1 ? "row was" : "rows were"} excluded.`;
    warning.hidden = false;
  }
}

async function loadDashboard() {
  try {
    const response = await fetch(API_URL, {headers: {Accept: "application/json"}, cache: "no-store"});
    if (!response.ok) throw new Error("request failed");
    renderDashboard(await response.json());
  } catch (_error) {
    document.getElementById("events").replaceChildren();
    showStatus("Upcoming predictions could not be loaded.", true);
  }
}

document.addEventListener("DOMContentLoaded", loadDashboard);
