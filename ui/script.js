const homeSelect = document.getElementById('home-team');
const awaySelect = document.getElementById('away-team');
const form = document.getElementById('predict-form');
const message = document.getElementById('message');
const results = document.getElementById('results');
const teamCount = document.getElementById('team-count');
const neutralVenue = document.getElementById('neutral-venue');

const resultTitle = document.getElementById('result-title');
const resultContext = document.getElementById('result-context');
const homeLabel = document.getElementById('home-label');
const awayLabel = document.getElementById('away-label');
const homeValue = document.getElementById('home-value');
const drawValue = document.getElementById('draw-value');
const awayValue = document.getElementById('away-value');
const homeBar = document.getElementById('home-bar');
const drawBar = document.getElementById('draw-bar');
const awayBar = document.getElementById('away-bar');
const eloGap = document.getElementById('elo-gap');
const formGap = document.getElementById('form-gap');

function setMessage(text, kind = '') {
  message.textContent = text;
  message.style.color = kind === 'error' ? '#ff8d9a' : 'var(--muted)';
}

function populateSelect(select, teams) {
  select.innerHTML = '<option value="">Select a country</option>';
  teams.forEach((team) => {
    const option = document.createElement('option');
    option.value = team;
    option.textContent = team;
    select.appendChild(option);
  });
}

function syncOppositionOptions() {
  const homeTeam = homeSelect.value;
  const awayTeam = awaySelect.value;

  Array.from(homeSelect.options).forEach((option) => {
    option.disabled = option.value && option.value === awayTeam;
  });

  Array.from(awaySelect.options).forEach((option) => {
    option.disabled = option.value && option.value === homeTeam;
  });
}

async function loadTeams() {
  try {
    const response = await fetch('/teams');
    const data = await response.json();
    const teams = data.teams || [];

    populateSelect(homeSelect, teams);
    populateSelect(awaySelect, teams);
    teamCount.textContent = String(teams.length);
    setMessage('Choose two countries to generate a prediction.');
  } catch (error) {
    setMessage('Unable to load country list.', 'error');
  }
}

function updateResults(data) {
  const homeTeam = data.home_team;
  const awayTeam = data.away_team;
  const probs = data.probabilities;

  resultTitle.textContent = `${homeTeam} vs ${awayTeam}`;
  resultContext.textContent = data.is_neutral ? 'Neutral venue' : `${homeTeam} have home advantage`;
  homeLabel.textContent = `${homeTeam} Win`;
  awayLabel.textContent = `${awayTeam} Win`;

  homeValue.textContent = `${probs.home_win.toFixed(1)}%`;
  drawValue.textContent = `${probs.draw.toFixed(1)}%`;
  awayValue.textContent = `${probs.away_win.toFixed(1)}%`;

  homeBar.style.width = `${probs.home_win}%`;
  drawBar.style.width = `${probs.draw}%`;
  awayBar.style.width = `${probs.away_win}%`;

  eloGap.textContent = `${data.insights.elo_advantage_points.toFixed(1)}`;
  formGap.textContent = `${data.insights.form_advantage_points.toFixed(2)}`;

  results.classList.remove('hidden');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const homeTeam = homeSelect.value;
  const awayTeam = awaySelect.value;

  if (!homeTeam || !awayTeam) {
    setMessage('Please choose both teams.', 'error');
    return;
  }

  if (homeTeam === awayTeam) {
    setMessage('Home and away teams must be different.', 'error');
    return;
  }

  setMessage('Calculating prediction...');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        home_team: homeTeam,
        away_team: awayTeam,
        is_neutral: neutralVenue.checked,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Prediction failed.');
    }

    updateResults(data);
    setMessage('Prediction updated.');
    syncOppositionOptions();
  } catch (error) {
    setMessage(error.message, 'error');
  }
});

homeSelect.addEventListener('change', syncOppositionOptions);
awaySelect.addEventListener('change', syncOppositionOptions);

loadTeams();
