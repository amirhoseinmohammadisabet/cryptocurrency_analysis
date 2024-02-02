function filterResults() {
    var nameInput = document.getElementById('nameInput').value;

    fetch(`/filter?name=${nameInput}`)
        .then(response => response.json())
        .then(data => displayResults(data));
}

function displayResults(results) {
    var resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.innerHTML = '';

    if (results.length === 0) {
        resultsContainer.innerHTML = '<p>No results found.</p>';
        return;
    }

    var table = '<table><tr><th>Name</th><th>Age</th><th>City</th></tr>';
    results.forEach(result => {
        table += `<tr><td>${result.Name}</td><td>${result.Age}</td><td>${result.City}</td></tr>`;
    });
    table += '</table>';

    resultsContainer.innerHTML = table;
}

    

function showName(name) {
    var nameElement = document.getElementById(name.toLowerCase() + 'Name');
    nameElement.classList.remove('hidden');
}

function hideName() {
    var names = document.querySelectorAll('.icon p');
    names.forEach(function (name) {
        name.classList.add('hidden');
    });
}