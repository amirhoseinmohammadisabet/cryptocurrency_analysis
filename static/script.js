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


document.addEventListener('DOMContentLoaded', function () {
    var chartTypeSelect = document.getElementById('chartType');
    var chartCanvas = document.getElementById('myChart');
    var ctx = chartCanvas.getContext('2d');

    // Initial chart type
    var selectedChartType = chartTypeSelect.value;

    // Initial data (you may customize this based on your CSV file structure)
    var data = {
        labels: ['Label 1', 'Label 2', 'Label 3'],
        datasets: [{
            label: 'Dataset',
            data: [10, 20, 30],
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
            ],
            borderWidth: 1,
        }],
    };

    // Create initial chart
    var myChart = new Chart(ctx, {
        type: selectedChartType,
        data: data,
    });

    // Update chart when the user changes the chart type
    chartTypeSelect.addEventListener('change', function () {
        selectedChartType = chartTypeSelect.value;
        updateChart();
    });

    function updateChart() {
        // Destroy the previous chart instance
        myChart.destroy();

        // Create a new chart based on the selected type
        myChart = new Chart(ctx, {
            type: selectedChartType,
            data: data,
        });
    }
});
    

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