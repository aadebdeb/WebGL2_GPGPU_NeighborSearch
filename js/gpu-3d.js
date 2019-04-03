(function() {

  const FILL_SCREEN_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in vec2 position;

void main(void) {
  gl_Position = vec4(position, 0.0, 1.0);
}
`

  const INITIALIZE_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

layout (location = 0) out vec4 o_position;
layout (location = 1) out vec4 o_velocity;

uniform vec2 u_randomSeed;

float random(vec2 x){
  return fract(sin(dot(x,vec2(12.9898, 78.233))) * 43758.5453);
}

void main(void) {
  o_position = vec4(
    sqrt(random(gl_FragCoord.xy * 0.013 + u_randomSeed + vec2(32.19, 27.51))),
    sqrt(random(gl_FragCoord.xy * 0.029 + u_randomSeed + vec2(19.56, 11.34))),
    sqrt(random(gl_FragCoord.xy * 0.035 + u_randomSeed + vec2(11.21, 35.89))),
    0.0
  );
  o_velocity = vec4(((vec3(
    random(gl_FragCoord.xy * 0.059 + u_randomSeed + vec2(27.31, 16.91)),
    random(gl_FragCoord.xy * 0.038 + u_randomSeed + vec2(25.95, 19.47)),
    random(gl_FragCoord.xy * 0.031 + u_randomSeed + vec2(31.71, 14.52))
  ) * 2.0 - 1.0) * 0.05), 0.0);
}
`

  const UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;

layout (location = 0) out vec3 o_position;
layout (location = 1) out vec3 o_velocity;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform float u_deltaTime;

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec3 position = texelFetch(u_positionTexture, coord, 0).xyz;
  vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;

  vec3 nextPosition = position + u_deltaTime * velocity;
  vec3 nextVelocity = velocity;

  if (nextPosition.x <= 0.0) {
    nextVelocity.x *= -1.0;
    nextPosition.x += u_deltaTime * nextVelocity.x;
  }
  if (nextPosition.x >= 1.0) {
    nextVelocity.x *= -1.0;
    nextPosition.x += u_deltaTime * nextVelocity.x;
  }
  if (nextPosition.y <= 0.0) {
    nextVelocity.y *= -1.0;
    nextPosition.y += u_deltaTime * nextVelocity.y;
  }
  if (nextPosition.y >= 1.0) {
    nextVelocity.y *= -1.0;
    nextPosition.y += u_deltaTime * nextVelocity.y;
  }
  if (nextPosition.z <= 0.001) {
    nextVelocity.z *= -1.0;
    nextPosition.z += u_deltaTime * nextVelocity.z;
  }
  if (nextPosition.z >= 0.999) {
    nextVelocity.z *= -1.0;
    nextPosition.z += u_deltaTime * nextVelocity.z;
  }

  o_position = nextPosition;
  o_velocity = nextVelocity;
}
`

  const RENDER_PARTICLE_VERTEX_SHADER_SOURCE =
`#version 300 es

precision highp isampler2D;
precision highp usampler2D;

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexNormal;

out vec3 v_color;
out vec3 v_normal;

uniform sampler2D u_positionTexture;
uniform usampler2D u_bucketTexture;
uniform isampler2D u_bucketReferrerTexture;
uniform float u_viewRadius;
uniform float u_maxValue;
uniform mat4 u_mvpMatrix;

float simulationSpace = 1.0;

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

float findNeighbors(vec3 position, ivec3 bucketPosition, ivec3 bucketNum, int particleTextureSizeX, int bucketReferrerTextureSizeX) {
  if (bucketPosition.x < 0 || bucketPosition.x >= bucketNum.x ||
      bucketPosition.y < 0 || bucketPosition.y >= bucketNum.y ||
      bucketPosition.z < 0 || bucketPosition.z >= bucketNum.z) {
    return 0.0;
  }
  int bucketIndex = bucketPosition.x + bucketNum.x * bucketPosition.y + (bucketNum.x * bucketNum.y) * bucketPosition.z;
  ivec2 coord = convertIndexToCoord(bucketIndex, bucketReferrerTextureSizeX);

  ivec2 bucketReferrer = ivec2(texelFetch(u_bucketReferrerTexture, coord, 0).xy);

  if (bucketReferrer.x == -1 || bucketReferrer.y == -1) {
    return 0.0;
  }

  float sum = 0.0;
  for (int i = bucketReferrer.x; i <= bucketReferrer.y; i++) {
    uvec2 bucket = texelFetch(u_bucketTexture, convertIndexToCoord(i, particleTextureSizeX), 0).xy;

    int particleIndex = int(bucket.y);
    if (gl_InstanceID == particleIndex) {
      continue;
    }
    ivec2 particleCoord = convertIndexToCoord(particleIndex, particleTextureSizeX);
    vec3 particlePos = texelFetch(u_positionTexture, particleCoord, 0).xyz;
    if (length(position - particlePos) < u_viewRadius) {
      sum += (u_viewRadius - length(position - particlePos)) / u_viewRadius;
    }
  }
  return sum;
}

vec3[7] HEAT_MAP_COLORS = vec3[7](
  vec3(0.5),
  vec3(0.0, 0.0, 1.0),
  vec3(0.0, 0.5, 0.5),
  vec3(0.0, 1.0, 0.0),
  vec3(0.5, 0.5, 0.0),
  vec3(0.8, 0.4, 0.0),
  vec3(1.0, 0.0, 0.0)
);

vec3 getHeatmapColor(float value, float maxValue) {
  float step = maxValue / 7.0;
  float i = min(value, maxValue - 0.0001) / step;
  return mix(HEAT_MAP_COLORS[int(i)], HEAT_MAP_COLORS[int(i) + 1], fract(i));
}

void main(void) {
  ivec2 coord = convertIndexToCoord(gl_InstanceID, textureSize(u_positionTexture, 0).x);
  vec3 position = texelFetch(u_positionTexture, coord, 0).xyz;

  v_normal = (u_mvpMatrix * vec4(vertexNormal, 0.0)).xyz;
  gl_Position = u_mvpMatrix * vec4(vertexPosition + (2.0 * position - 1.0) * 500.0, 1.0);

  int particleTextureSizeX = textureSize(u_positionTexture, 0).x;
  int bucketReferrerTextureSizeX = textureSize(u_bucketReferrerTexture, 0).x;

  vec3 bucketPosition = position / (2.0 * u_viewRadius);
  int xOffset = fract(bucketPosition.x) < 0.5 ? -1 : 1;
  int yOffset = fract(bucketPosition.y) < 0.5 ? -1 : 1;
  int zOffset = fract(bucketPosition.z) < 0.5 ? -1 : 1;

  ivec3 bucketPosition000 = ivec3(bucketPosition);
  ivec3 bucketPosition100 = bucketPosition000 + ivec3(xOffset, 0, 0);
  ivec3 bucketPosition010 = bucketPosition000 + ivec3(0, yOffset, 0);
  ivec3 bucketPosition110 = bucketPosition000 + ivec3(xOffset, yOffset, 0);
  ivec3 bucketPosition001 = bucketPosition000 + ivec3(0, 0, zOffset);
  ivec3 bucketPosition101 = bucketPosition000 + ivec3(xOffset, 0, zOffset);
  ivec3 bucketPosition011 = bucketPosition000 + ivec3(0, yOffset, zOffset);
  ivec3 bucketPosition111 = bucketPosition000 + ivec3(xOffset, yOffset, zOffset);

  ivec3 bucketNum = ivec3(simulationSpace / (2.0 * u_viewRadius)) + 1;

  float sum = 0.0;
  sum += findNeighbors(position, bucketPosition000, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  sum += findNeighbors(position, bucketPosition100, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  sum += findNeighbors(position, bucketPosition010, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  sum += findNeighbors(position, bucketPosition110, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  sum += findNeighbors(position, bucketPosition001, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  sum += findNeighbors(position, bucketPosition101, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  sum += findNeighbors(position, bucketPosition011, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);
  sum += findNeighbors(position, bucketPosition111, bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX);


  v_color = getHeatmapColor(sum, u_maxValue);
}
`

  const RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_color;
in vec3 v_normal;

out vec4 o_color;

vec3 LightDir = normalize(vec3(1.0, 1.0, -1.0));

void main(void) {
  vec3 normal = normalize(v_normal);
  float dotNL = dot(normal, LightDir);
  o_color = vec4(v_color * smoothstep(-1.0, 1.0, dotNL), 1.0);
}
`

const INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out uvec2 o_particle;

uniform sampler2D u_positionTexture;
uniform float u_viewRadius;

float simulationSpace = 1.0;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

uint getBucketIndex(vec3 position) {
  uvec3 bucketCoord = uvec3(position / (2.0 * u_viewRadius));
  uvec3 bucketNum = uvec3(simulationSpace / (2.0 * u_viewRadius)) + 1u;
  return bucketCoord.x + bucketCoord.y * bucketNum.x + bucketCoord.z * (bucketNum.x * bucketNum.y);
}

void main(void) {
  vec3 position = texelFetch(u_positionTexture, ivec2(gl_FragCoord.xy), 0).xyz;
  uint positionTextureSizeX = uint(textureSize(u_positionTexture, 0).x);
  uint particleIndex = convertCoordToIndex(uvec2(gl_FragCoord.xy), positionTextureSizeX);
  uint bucketIndex = getBucketIndex(position);
  o_particle = uvec2(bucketIndex, particleIndex);
}

`

  const SWAP_BUCKET_INDEX_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;
precision highp usampler2D;

out uvec3 o_bucket;

uniform usampler2D u_bucketTexture;
uniform uint u_size;
uniform uint u_blockStep;
uniform uint u_subBlockStep;

uint convertCoordToIndex(uvec2 coord, uint sizeX) {
  return coord.x + sizeX * coord.y;
}

uvec2 convertIndexToCoord(uint index, uint sizeX) {
  return uvec2(index % sizeX, index / sizeX);
}

void main(void) {
  uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy), u_size);
  uint d = 1u << (u_blockStep - u_subBlockStep);

  bool up = ((index >> u_blockStep) & 2u) == 0u;

  uint targetIndex;
  bool first = (index & d) == 0u;
  if (first) {
    targetIndex = index | d;
  } else {
    targetIndex = index & ~d;
    up = !up;
  }

  uvec3 a = texelFetch(u_bucketTexture, ivec2(gl_FragCoord.xy), 0).xyz;
  uvec3 b = texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(targetIndex, u_size)), 0).xyz;

  if (a.x == b.x || (a.x >= b.x) == up) {
    o_bucket = b;
  } else {
    o_bucket = a;
  }
}

`;

  const INITIALIZE_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;
precision highp usampler2D;

out ivec2 o_referrer;

uniform ivec2 u_bucketReferrerTextureSize;
uniform usampler2D u_bucketTexture;
uniform float u_viewRadius;
uniform int u_particleNumN;

float simulationSpace = 1.0;

int convertCoordToIndex(ivec2 coord, int sizeX) {
  return coord.x + sizeX * coord.y;
}

ivec2 convertIndexToCoord(int index, int sizeX) {
  return ivec2(index % sizeX, index / sizeX);
}

int getBucketIndex(int particleIndex, int particleTextureSizeX) {
  return int(texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(particleIndex, particleTextureSizeX)), 0).x);
}


int binarySearchMinIndex(int target, int from, int to, int particleTextureSizeX) {
  for (int i = 0; i < u_particleNumN + 1; i++) {
    int middle = from + (to - from) / 2;
    int bucketIndex = getBucketIndex(middle, particleTextureSizeX);
    if (bucketIndex < target) {
      from = middle + 1;
    } else {
      to = middle;
    }
    if (from == to) {
      if (getBucketIndex(from, particleTextureSizeX) == target) {
        return from;
      } else {
        return -1;
      }
    }
  }
  return -1;
}

int binarySearchMaxIndex(int target, int from, int to, int particleTextureSizeX) {
  for (int i = 0; i < u_particleNumN + 1; i++) {
    int middle = from + (to - from) / 2 + 1;
    int bucketIndex = getBucketIndex(middle, particleTextureSizeX);
    if (bucketIndex > target) {
      to = middle - 1;
    } else {
      from = middle;
    }
    if (from == to) {
      if (getBucketIndex(from, particleTextureSizeX) == target) {
        return from;
      } else {
        return -1;
      }
    }
  }
  return -1;
}

ivec2 binarySearchRange(int target, int from, int to) {
  int particleTextureSizeX = textureSize(u_bucketTexture, 0).x;
  from =  binarySearchMinIndex(target, from, to, particleTextureSizeX);
  to = from == -1 ? -1 : binarySearchMaxIndex(target, from, to, particleTextureSizeX);
  return ivec2(from, to);
}

void main(void) {
  int bucketIndex = convertCoordToIndex(ivec2(gl_FragCoord.xy), u_bucketReferrerTextureSize.x);
  ivec3 bucketNum = ivec3(simulationSpace / (2.0 * u_viewRadius)) + 1;
  int maxBucketIndex = bucketNum.x * bucketNum.y * bucketNum.z;

  if (bucketIndex >= maxBucketIndex) {
    o_referrer = ivec2(-1, -1);
    return;
  }

  ivec2 particleTextureSize = textureSize(u_bucketTexture, 0);
  int particleNum = particleTextureSize.x * particleTextureSize.y;

  o_referrer = binarySearchRange(bucketIndex, 0, particleNum - 1);
}
`

  const DEBUG_BITONICSORT_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;
precision highp usampler2D;

out vec4 o_color;

uniform usampler2D u_bucketTexture;
uniform vec2 u_canvasSize;
uniform uint u_bucketNum;

void main(void) {
  uvec2 bucket = texture(u_bucketTexture, gl_FragCoord.xy / u_canvasSize).xy;
  // o_color = vec4(vec3(float(bucket.x) / float(u_bucketNum)), 1.0);
  o_color = vec4(vec3(float(bucket.y) / float(16384 - 1)), 1.0);
}
`

  const DEBUG_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;
precision highp isampler2D;

out vec4 o_color;

uniform isampler2D u_bucketReferrerTexture;
uniform vec2 u_canvasSize;
uniform uint u_particleNum;

void main(void) {
  ivec2 bucketReferrer = texture(u_bucketReferrerTexture, gl_FragCoord.xy / u_canvasSize).xy;
  if (bucketReferrer.x == -1) {
    o_color = vec4(1.0, 0.0, 0.0, 1.0);
  } else {
    o_color = vec4(vec3(float(bucketReferrer.x) / float(u_particleNum)), 1.0);
  }
}
`

  const VERTICES_POSITION = new Float32Array([
    -1.0, -1.0,
    1.0, -1.0,
    -1.0,  1.0,
    1.0,  1.0
  ]);

  const VERTICES_INDEX = new Int16Array([
    0, 1, 2,
    3, 2, 1
  ]);


  function createInitializeParticleProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, INITIALIZE_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createUpdateParticleProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createRenderParticleProgram(gl) {
    const vertexShader = createShader(gl, RENDER_PARTICLE_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createInitializeBucketProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createSwapBucketIndexProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, SWAP_BUCKET_INDEX_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createInitializeBucketReferrerProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, INITIALIZE_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createDebugBitonicSortProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, DEBUG_BITONICSORT_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createDebugBucketReferrerProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, DEBUG_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createParticleFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const positionTexture = createTexture(gl, size, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const velocityTexture = createTexture(gl, size, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, velocityTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, velocityTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      positionTexture: positionTexture,
      velocityTexture: velocityTexture
    };
  }

  function createBucketFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const bucketTexture = createTexture(gl, size, gl.RG32UI, gl.RG_INTEGER, gl.UNSIGNED_INT);
    gl.bindTexture(gl.TEXTURE_2D, bucketTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bucketTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      bucketTexture: bucketTexture
    };
  }

  function createBucketReferrerFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const bucketReferrerTexture = createTexture(gl, size, gl.RG32I, gl.RG_INTEGER, gl.INT);
    gl.bindTexture(gl.TEXTURE_2D, bucketReferrerTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bucketReferrerTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      bucketReferrerTexture: bucketReferrerTexture
    };
  }

  const canvas = document.getElementById('canvas');
  const resizeCanvas = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);

  const initializeParticleProgram = createInitializeParticleProgram(gl);
  const updateParticleProgram = createUpdateParticleProgram(gl);
  const renderParticleProgram = createRenderParticleProgram(gl);
  const initializeBucketProgram = createInitializeBucketProgram(gl);
  const swapBucketIndexProgram = createSwapBucketIndexProgram(gl);
  const initializeBucketReferrerProgram = createInitializeBucketReferrerProgram(gl);
  const debugBitonicSortProgram = createDebugBitonicSortProgram(gl);
  const debugBucketReferrerProgram = createDebugBucketReferrerProgram(gl);

  const initializeParticleUniforms = getUniformLocations(gl, initializeParticleProgram, ['u_randomSeed']);
  const updateParticleUniforms = getUniformLocations(gl, updateParticleProgram, ['u_positionTexture', 'u_velocityTexture', 'u_deltaTime']);
  const renderParticleUniforms = getUniformLocations(gl, renderParticleProgram, ['u_positionTexture', 'u_bucketTexture', 'u_bucketReferrerTexture', 'u_viewRadius', 'u_maxValue', 'u_mvpMatrix']);
  const initializeBucketUniforms = getUniformLocations(gl, initializeBucketProgram, ['u_positionTexture', 'u_viewRadius']);
  const swapBucketIndexUniforms = getUniformLocations(gl, swapBucketIndexProgram, ['u_bucketTexture', 'u_size', 'u_blockStep', 'u_subBlockStep']);
  const initializeBucketReferrerUniforms = getUniformLocations(gl, initializeBucketReferrerProgram, ['u_bucketTexture', 'u_viewRadius', 'u_bucketReferrerTextureSize', 'u_particleNumN']);
  const debugBitonicSortUniforms = getUniformLocations(gl, debugBitonicSortProgram, ['u_bucketTexture', 'u_bucketNum', 'u_canvasSize']);
  const debugBucketReferrerUniforms = getUniformLocations(gl, debugBucketReferrerProgram, ['u_bucketReferrerTexture', 'u_canvasSize', 'u_particleNum'])

  const fillScreenVao = createVao(gl,
    [{
      buffer: createVbo(gl, VERTICES_POSITION),
      size: 2,
      index: 0
    }],
    createIbo(gl, VERTICES_INDEX)
  );

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const gui = new dat.GUI();
  const data = {
    'N': 5,
    'particle num': 2 ** (5 * 2),
    'view radius': 0.1,
    'max value': 300,
    sphere: {
      'radius': 5.0,
      'theta segment': 16,
      'phi segment': 8
    },
    camera: {
      'angle': -45.0,
      'distance': 1000.0,
      'height': 1000.0
    },
    'reset': () => reset()
  };
  gui.add(data, 'N', 1, 8).step(1).onChange((v) => {
    data['particle num'] = 2 ** (v * 2);
  });
  gui.add(data, 'particle num').listen();
  gui.add(data, 'view radius', 0.05, 0.25);
  gui.add(data, 'max value', 0, 1000);
  const guiSphere = gui.addFolder('sphere');
  guiSphere.add(data.sphere, 'radius', 1.0, 20.0);
  guiSphere.add(data.sphere, 'theta segment', 4, 64).step(1);
  guiSphere.add(data.sphere, 'phi segment', 4, 64).step(1);
  const guiCamera = gui.addFolder('camera');
  guiCamera.add(data.camera, 'angle', -180, 180);
  guiCamera.add(data.camera, 'distance', 50.0, 3000.0);
  guiCamera.add(data.camera, 'height', -3000.0, 3000.0);
  gui.add(data, 'reset');

  let requestId = null;
  function reset() {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const sphere = createSphere(data.sphere['radius'], data.sphere['theta segment'], data.sphere['phi segment']);
    const sphereIbo = createIbo(gl, sphere.indices);
    const vertexPositionVbo = createVbo(gl, sphere.positions);
    const vertexNormalVbo = createVbo(gl, sphere.normals);

    const particleTextureSizeN = data['N'];
    const particleTextureSize = 2 ** particleTextureSizeN;
    const particleNumN = particleTextureSizeN * 2;
    const particleNum = particleTextureSize * particleTextureSize;
  
    const viewRadius = data['view radius'];
    const bucketSize = Math.ceil(1.0 / (2.0 * viewRadius));
    const bucketNum = bucketSize * bucketSize * bucketSize;
    let bucketReferrerTextureSize;
    for (let i = 0; ; i++) {
      bucketReferrerTextureSize = 2 ** i;
      if (bucketReferrerTextureSize * bucketReferrerTextureSize > bucketNum) {
        break;
      }
    }
  
    const particleVao = gl.createVertexArray();
    gl.bindVertexArray(particleVao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sphereIbo);
    [vertexPositionVbo, vertexNormalVbo].forEach((vbo, i) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 0, 0);
    });
    gl.bindVertexArray(null);
  
    let particleFbObjR = createParticleFramebuffer(gl, particleTextureSize);
    let particleFbObjW = createParticleFramebuffer(gl, particleTextureSize);
    const swapParticleFbObj = function() {
      const tmp = particleFbObjR;
      particleFbObjR = particleFbObjW;
      particleFbObjW = tmp;
    };
  
    let bucketFbObjR = createBucketFramebuffer(gl, particleTextureSize);
    let bucketFbObjW = createBucketFramebuffer(gl, particleTextureSize);
    const swapBucketFbObj = function() {
      const tmp = bucketFbObjR;
      bucketFbObjR = bucketFbObjW;
      bucketFbObjW = tmp;
    }
  
    const bucketReferrerFbObj = createBucketReferrerFramebuffer(gl, bucketReferrerTextureSize);
  
    const initializeParticles = function() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, particleFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
  
      gl.useProgram(initializeParticleProgram);
      gl.uniform2f(initializeParticleUniforms['u_randomSeed'], Math.random() * 100.0, Math.random() * 100.0);
  
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
  
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapParticleFbObj();
    };
  
    const initializeBucket = function() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bucketFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
  
      gl.useProgram(initializeBucketProgram);
      setTextureAsUniform(gl, 0, particleFbObjR.positionTexture, initializeBucketUniforms['u_positionTexture']);
      gl.uniform1f(initializeBucketUniforms['u_viewRadius'], viewRadius);
  
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
  
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapBucketFbObj();
    }
  
    const swapBucketIndex = function(i, j) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bucketFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
  
      gl.useProgram(swapBucketIndexProgram);
      setTextureAsUniform(gl, 0, bucketFbObjR.bucketTexture, swapBucketIndexUniforms['u_bucketTexture']);
      gl.uniform1ui(swapBucketIndexUniforms['u_size'], particleTextureSize, particleTextureSize);
      gl.uniform1ui(swapBucketIndexUniforms['u_blockStep'], i);
      gl.uniform1ui(swapBucketIndexUniforms['u_subBlockStep'], j);
  
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
  
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapBucketFbObj();
    }
  
    const initializeBucketRefrrer = function() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bucketReferrerFbObj.framebuffer);
      gl.viewport(0.0, 0.0, bucketReferrerTextureSize, bucketReferrerTextureSize);
  
      gl.useProgram(initializeBucketReferrerProgram);
      setTextureAsUniform(gl, 0, bucketFbObjR.bucketTexture, initializeBucketReferrerUniforms['u_bucketTexture']);
      gl.uniform1f(initializeBucketReferrerUniforms['u_viewRadius'], viewRadius);
      gl.uniform1i(initializeBucketReferrerUniforms['u_particleNumN'], particleNumN);
      gl.uniform2i(initializeBucketReferrerUniforms['u_bucketReferrerTextureSize'], bucketReferrerTextureSize, bucketReferrerTextureSize);
  
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
  
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
  
    const constructBuckets = function() {
      initializeBucket();
  
      // sort by bitonic sort
      for (let i = 0; i < particleNumN; i++) {
        for (let j = 0; j <= i; j++) {
          swapBucketIndex(i, j);
        }
      }
      
      initializeBucketRefrrer();
    }
  
    const updateParticles = function(deltaTime) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, particleFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
  
      gl.useProgram(updateParticleProgram);
      setTextureAsUniform(gl, 0, particleFbObjR.positionTexture, updateParticleUniforms['u_positionTexture']);
      setTextureAsUniform(gl, 1, particleFbObjR.velocityTexture, updateParticleUniforms['u_velocityTexture']);
      gl.uniform1f(updateParticleUniforms['u_deltaTime'], deltaTime);
  
      gl.bindVertexArray(fillScreenVao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
  
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapParticleFbObj();
  
      constructBuckets();
    };
  
  
    initializeParticles();
  
    gl.clearColor(0.2, 0.2, 0.2, 1.0);
    let previousTime = performance.now();
    const render = function() {
      stats.update();
  
      const currentTime = performance.now();
      const deltaTime = Math.min(0.05, (currentTime - previousTime) * 0.001);
      previousTime = currentTime;

      const cameraRadian = Math.PI * data.camera['angle'] / 180.0;
      const viewMatrix = Matrix4.inverse(Matrix4.lookAt(
        new Vector3(data.camera['distance'] * Math.cos(cameraRadian), data.camera['height'], data.camera['distance'] * Math.sin(cameraRadian)),
        Vector3.zero, new Vector3(0.0, 1.0, 0.0)
      ));
      const projectionMatrix = Matrix4.perspective(canvas.width / canvas.height, 60, 0.01, 10000.0);
      const mvpMatrix = Matrix4.mul(viewMatrix, projectionMatrix);

      updateParticles(deltaTime);
  
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      gl.useProgram(renderParticleProgram);
      setTextureAsUniform(gl, 0, particleFbObjR.positionTexture, renderParticleUniforms['u_positionTexture']);
      setTextureAsUniform(gl, 1, bucketFbObjR.bucketTexture, renderParticleUniforms['u_bucketTexture']);
      setTextureAsUniform(gl, 2, bucketReferrerFbObj.bucketReferrerTexture, renderParticleUniforms['u_bucketReferrerTexture']);
      gl.uniform2f(renderParticleUniforms['u_canvasSize'], canvas.width, canvas.height);
      gl.uniform1f(renderParticleUniforms['u_viewRadius'], viewRadius);
      gl.uniform1f(renderParticleUniforms['u_maxValue'], data['max value']);
      gl.uniformMatrix4fv(renderParticleUniforms['u_mvpMatrix'], false, mvpMatrix.elements);

      gl.bindVertexArray(particleVao);
      gl.drawElementsInstanced(gl.TRIANGLES, sphere.indices.length, gl.UNSIGNED_SHORT, 0, particleNum);
      gl.bindVertexArray(null);
  
  
      // code to render bucketTexture for debug
      // gl.useProgram(debugBitonicSortProgram);
      // setTextureAsUniform(gl, 0, bucketFbObjR.bucketTexture, debugBitonicSortUniforms['u_bucketTexture']);
      // gl.uniform1ui(debugBitonicSortUniforms['u_bucketNum'], bucketSize * bucketSize);
      // gl.uniform2f(debugBitonicSortUniforms['u_canvasSize'], canvas.width, canvas.height);
      // gl.bindVertexArray(fillScreenVao);
      // gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      // gl.bindVertexArray(null);
  
      // code to render bucketReferrerTexture for debug
      // gl.useProgram(debugBucketReferrerProgram);
      // setTextureAsUniform(gl, 0, bucketReferrerFbObj.bucketReferrerTexture, debugBucketReferrerUniforms['u_bucketReferrerTexture']);
      // gl.uniform1ui(debugBucketReferrerUniforms['u_particleNum'], particleNum);
      // gl.uniform2f(debugBucketReferrerUniforms['u_canvasSize'], canvas.width, canvas.height);
      // gl.bindVertexArray(fillScreenVao);
      // gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      // gl.bindVertexArray(null);
  
      requestId = requestAnimationFrame(render);
    };
    render();
  };
  reset();


}());