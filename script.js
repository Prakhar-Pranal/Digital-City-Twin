$(document).ready(function(){
    $(window).scroll(function(){
        // Sticky navbar on scroll
        if(this.scrollY > 20){
            $('.navbar').addClass("sticky");
        } else {
            $('.navbar').removeClass("sticky");
        }
        
        // Scroll-up button show/hide
        if(this.scrollY > 500){
            $('.scroll-up-btn').addClass("show");
        } else {
            $('.scroll-up-btn').removeClass("show");
        }
    });

    // Scroll-up button action
    $('.scroll-up-btn').click(function(){
        $('html').animate({scrollTop: 0});
        $('html').css("scrollBehavior", "auto");
    });

    $('.navbar .menu li a').click(function(){
        $('html').css("scrollBehavior", "smooth");
    });

    // Toggle navbar menu
    $('.menu-btn').click(function(){
        $('.navbar .menu').toggleClass("active");
        $('.menu-btn i').toggleClass("active");
    });

    // Typing animation effect
    var typed = new Typed(".typing", {
        strings: ["Tech Enthusiasts", "Creators", "Solvers", " Engineers"],
        typeSpeed: 50,
        backSpeed: 90,
        loop: true
    });

    // Owl Carousel initialization
    $('.carousel').owlCarousel({
        margin: 20,
        loop: true,
        autoplay: true,
        autoplayTimeOut: 2000,
        autoplayHoverPause: true,
        responsive: {
            0:{ items: 1, nav: false },
            600:{ items: 2, nav: false },
            1000:{ items: 3, nav: false }
        }
    });

    // Simulated Smart City Data (Real-time updates)
    function updateStats() {
        document.getElementById("traffic").innerText = `${Math.floor(Math.random() * 100)}%`;
        document.getElementById("pollution").innerText = `${Math.floor(Math.random() * 300)} AQI`;
        document.getElementById("energy").innerText = `${Math.floor(Math.random() * 500)} kWh`;
    }
    setInterval(updateStats, 3000);
    updateStats();
});

// --- THREE.JS CITY SIMULATION ---
// Select the city container div
const cityContainer = document.getElementById("city-container");
cityContainer.style.width = "1300px";
cityContainer.style.height = "750px";

cityContainer.style.overflow = "hidden";

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, 800 / 600, 0.1, 900);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(800, 600);
cityContainer.appendChild(renderer.domElement);
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('city-container').appendChild(renderer.domElement);

// Lighting for Night Scene
const ambientLight = new THREE.AmbientLight(0x222222); // Dim ambient light for night effect
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5); // Lower intensity for night
directionalLight.position.set(10, 20, 10);
scene.add(directionalLight);

const hemiLight = new THREE.HemisphereLight(0x222233, 0x000000, 0.5); // Darker sky reflection
scene.add(hemiLight);

// Sky with Night Effect and Moon
const skyGeometry = new THREE.SphereGeometry(100, 32, 32);
const skyMaterial = new THREE.MeshBasicMaterial({ color: 0x0a0a32, side: THREE.BackSide }); // Dark blue night sky
const sky = new THREE.Mesh(skyGeometry, skyMaterial);
sky.position.y = 10;
scene.add(sky);

// Moon with glow effect
const moonGeometry = new THREE.SphereGeometry(3, 32, 32);
const moonMaterial = new THREE.MeshStandardMaterial({ color: 0xffffcc, emissive: 0xffffcc, emissiveIntensity: 1 });
const moon = new THREE.Mesh(moonGeometry, moonMaterial);
moon.position.set(-20, 20, 1);
scene.add(moon);

// Traffic Signals for Each Side
const createTrafficSignal = (x, z) => {
  const poleGeometry = new THREE.CylinderGeometry(0.1, 0.1, 3, 16);
  const poleMaterial = new THREE.MeshPhongMaterial({ color: 0x333333 });
  const pole = new THREE.Mesh(poleGeometry, poleMaterial);
  pole.position.set(x, 1.5, z);
  scene.add(pole);

  const lightBoxGeometry = new THREE.BoxGeometry(0.5, 1.5, 0.5);
  const lightBoxMaterial = new THREE.MeshPhongMaterial({ color: 0x222222 });
  const lightBox = new THREE.Mesh(lightBoxGeometry, lightBoxMaterial);
  lightBox.position.set(x, 2.75, z);
  scene.add(lightBox);

  const lightColors = [0xff0000, 0xffff00, 0x00ff00];
  for (let i = 0; i < 3; i++) {
    const lightGeometry = new THREE.SphereGeometry(0.2, 16, 16);
    const lightMaterial = new THREE.MeshPhongMaterial({ color: lightColors[i] });
    const light = new THREE.Mesh(lightGeometry, lightMaterial);
    light.position.set(x, 3.1 - i * 0.5, z + 0.3);
    scene.add(light);
  }
};

createTrafficSignal(4, 0);
createTrafficSignal(-4, 0);
createTrafficSignal(0, 4);
createTrafficSignal(0, -4);

// Buildings with small roads
const buildingGeometry = new THREE.BoxGeometry(1, 1, 1);
const gridSize = 15;
const spacing = 2.5;
const roadWidth = 5;
const smallRoadWidth = 1.2;
const roadCenter = Math.floor(gridSize / 2);

for (let i = 0; i < gridSize; i++) {
  for (let j = 0; j < gridSize; j++) {
    const x = i * spacing - (gridSize * spacing) / 2;
    const z = j * spacing - (gridSize * spacing) / 2;

    if ((x < -4 && z > 4) || Math.abs(x) < 4 || Math.abs(z) < 4 || i === roadCenter || j === roadCenter) continue;

    const buildingMaterial = new THREE.MeshPhongMaterial({ color: Math.random() * 0x444444 }); // Darker buildings for night
    const building = new THREE.Mesh(buildingGeometry, buildingMaterial);
    building.scale.y = Math.random() * 4 + 1;
    building.position.set(x, building.scale.y / 2, z);
    scene.add(building);
  }
}

// City Hall with space
const cityHallGeometry = new THREE.BoxGeometry(6, 5, 4);
const cityHallMaterial = new THREE.MeshPhongMaterial({ color: 0x222266 }); // Darker shade for city hall
const cityHall = new THREE.Mesh(cityHallGeometry, cityHallMaterial);
cityHall.position.set(-8, 2.5, 8);
scene.add(cityHall);

// Roads
const roadGeometry = new THREE.BoxGeometry(gridSize * spacing, 0.1, roadWidth);
const roadMaterial = new THREE.MeshPhongMaterial({ color: 0x000000 });

const roads = [];

for (let i = 0; i < gridSize; i++) {
  const x = i * spacing - (gridSize * spacing) / 2;
  
  const roadX = new THREE.Mesh(roadGeometry, roadMaterial);
  roadX.position.set(x, 0.05, 0);
  scene.add(roadX);
  roads.push({ x, z: 0 });

  const roadZ = new THREE.Mesh(roadGeometry, roadMaterial);
  roadZ.rotation.y = Math.PI / 2;
  roadZ.position.set(0, 0.05, x);
  scene.add(roadZ);
  roads.push({ x: 0, z: x });
}

// Cars on roads
const carGeometry = new THREE.BoxGeometry(1, 0.5, 2);
const carMaterial = new THREE.MeshPhongMaterial({ color: 0xff0000 });

for (let i = 0; i < 10; i++) {
  const road = roads[Math.floor(Math.random() * roads.length)];
  const car = new THREE.Mesh(carGeometry, carMaterial);
  car.position.set(road.x, 0.3, road.z);
  scene.add(car);
}

// Camera positioning
camera.position.set(18,32,32);
camera.lookAt(0, 0, 0);

// Mouse controls
let isDragging = false;
let previousMousePosition = { x: 0, y: 0 };

document.addEventListener('mousedown', () => (isDragging = true));
document.addEventListener('mouseup', () => (isDragging = false));
document.addEventListener('mousemove', (event) => {
  if (isDragging) {
    const deltaX = (event.clientX - previousMousePosition.x) * 0.05;
    const deltaY = (event.clientY - previousMousePosition.y) * 0.05;
    

    camera.position.x -= deltaX;
    camera.position.y = Math.max(3, camera.position.y + deltaY);
  
    camera.lookAt(0, 0, 0);
  }

  previousMousePosition.x = event.clientX;
  previousMousePosition.y = event.clientY;
});
// Animation loop
const animate = () => {
  requestAnimationFrame(animate);
//   cars.forEach(car => {
//     if (car.axis === "x") {
//         car.mesh.position.x += car.direction * 0.02;
//         if (Math.abs(car.mesh.position.x) > 3) car.direction *= -1;
//     } else {
//         car.mesh.position.z += car.direction * 0.02;
//         if (Math.abs(car.mesh.position.z) > 3) car.direction *= -1;
//     }
// });
  renderer.render(scene, camera);
};
animate();




// Window resize handling
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});``