(function() {
  function createShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) + source);
    }
    return shader;
  }

  function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
  }

  function createVbo(gl, array) {
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vbo;
  }

  function createIbo(gl, array) {
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, array, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    return ibo;
  }

  function createVao(gl, vboObjs, ibo) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    if (ibo !== undefined) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    }
    vboObjs.forEach((vboObj) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, vboObj.buffer);
      gl.enableVertexAttribArray(vboObj.index);
      gl.vertexAttribPointer(vboObj.index, vboObj.size, gl.FLOAT, false, 0, 0);
    });
    gl.bindVertexArray(null);
    if (ibo !== undefined) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vao;
  }

  function createTexture(gl, size, internalFormat, format, type) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, size, size, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }

  function getUniformLocations(gl, program, keys) {
    const locations = {};
    keys.forEach(key => {
        locations[key] = gl.getUniformLocation(program, key);
    });
    return locations;
  }

  function setTextureAsUniform(gl, index, texture, location) {
    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(location, index);
  }

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

layout (location = 0) out vec2 o_position;
layout (location = 1) out vec2 o_velocity;

uniform vec2 u_randomSeed;

float random(vec2 x){
  return fract(sin(dot(x,vec2(12.9898, 78.233))) * 43758.5453);
}

void main(void) {
  o_position = vec2(
    pow(random(gl_FragCoord.xy * 0.013 + u_randomSeed + vec2(32.19, 27.51)), 0.5),
    pow(random(gl_FragCoord.xy * 0.029 + u_randomSeed + vec2(19.56, 11.34)), 0.5)
  );
  // o_position = vec2(
  //   random(gl_FragCoord.xy * 0.013 + u_randomSeed + vec2(32.19, 27.51)),
  //   random(gl_FragCoord.xy * 0.029 + u_randomSeed + vec2(19.56, 11.34))
  // );
  o_velocity = (vec2(
    random(gl_FragCoord.xy * 0.059 + u_randomSeed + vec2(27.31, 16.91)),
    random(gl_FragCoord.xy * 0.038 + u_randomSeed + vec2(25.95, 19.47))
  ) * 2.0 - 1.0) * 0.2;
}
`

  const UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;

layout (location = 0) out vec2 o_position;
layout (location = 1) out vec2 o_velocity;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform float u_deltaTime;

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;

  vec2 nextPosition = position + u_deltaTime * velocity;
  vec2 nextVelocity = velocity;

  if (nextPosition.x <= 0.001) {
    nextVelocity.x *= -1.0;
    nextPosition.x += u_deltaTime * nextVelocity.x;
  }
  if (nextPosition.x >= 0.999) {
    nextVelocity.x *= -1.0;
    nextPosition.x += u_deltaTime * nextVelocity.x;
  }
  if (nextPosition.y <= 0.001) {
    nextVelocity.y *= -1.0;
    nextPosition.y += u_deltaTime * nextVelocity.y;
  }
  if (nextPosition.y >= 0.999) {
    nextVelocity.y *= -1.0;
    nextPosition.y += u_deltaTime * nextVelocity.y;
  }

  o_position = nextPosition;
  o_velocity = nextVelocity;
}
`

  const RENDER_PARTICLE_VERTEX_SHADER_SOURCE =
`#version 300 es

precision highp isampler2D;
precision highp usampler2D;

layout (location = 0) in float o_index;

out vec3 v_color;

uniform sampler2D u_positionTexture;
uniform usampler2D u_bucketTexture;
uniform isampler2D u_bucketReferrerTexture;
uniform vec2 u_canvasSize;
uniform float u_viewRadius;

float simulationSpace = 1.0;

uint convertBucketPositionToIndex(uvec2 bucketPosition) {
  uvec2 bucketNum = uvec2(simulationSpace / u_viewRadius) + 1u;
  return bucketPosition.x + bucketPosition.y * bucketNum.x;
}

ivec2 convertIndexToCoord(int index, ivec2 size) {
  return ivec2(index % size.x, index / size.x);
}

ivec2 convertBucketIndexToCoord(int bucketIndex) {
  ivec2 size = textureSize(u_bucketReferrerTexture, 0);
  return ivec2(bucketIndex % size.x, bucketIndex / size.x);
}

ivec2 convertParticleIndexToCoord(int particleIndex) {
  ivec2 size = textureSize(u_positionTexture, 0);
  return ivec2(particleIndex % size.x, particleIndex / size.x);
}


float findNeighbors(vec2 position, ivec2 bucketPosition, ivec2 bucketNum) {
  if (bucketPosition.x < 0 || bucketPosition.x >= bucketNum.x || bucketPosition.y < 0 || bucketPosition.y >= bucketNum.y) {
    return 0.0;
  }
  int bucketIndex = bucketPosition.x + bucketNum.x * bucketPosition.y;
  ivec2 coord = convertBucketIndexToCoord(bucketIndex);

  ivec2 bucketReferrer = ivec2(texelFetch(u_bucketReferrerTexture, coord, 0).xy);

  if (bucketReferrer.x == -1 || bucketReferrer.y == -1) {
    return 0.0;
  }

  float sum = 0.0;
  for (int i = bucketReferrer.x; i <= bucketReferrer.y; i++) {
    uvec2 bucket = texelFetch(u_bucketTexture, convertParticleIndexToCoord(i), 0).xy;

    int particleIndex = int(bucket.y);
    if (int(o_index) == particleIndex) {
      continue;
    }
    ivec2 particleCoord = convertParticleIndexToCoord(particleIndex);
    vec2 particlePos = texelFetch(u_positionTexture, particleCoord, 0).xy;
    if (length(position - particlePos) < u_viewRadius) {
      sum += (u_viewRadius - length(position - particlePos)) / u_viewRadius;
    }
  }
  return sum;
}

void main(void) {
  vec2 scale = min(u_canvasSize.x, u_canvasSize.y) / u_canvasSize;
  ivec2 coord = convertIndexToCoord(int(o_index), textureSize(u_positionTexture, 0));
  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  gl_Position = vec4(scale * (position * 2.0 - 1.0), 0.0, 1.0);
  gl_PointSize = 5.0;


  vec2 bucketPosition = position / u_viewRadius;
  int xOffset = fract(bucketPosition.x) < 0.5 ? -1 : 1;
  int yOffset = fract(bucketPosition.y) < 0.5 ? -1 : 1;

  ivec2 bucketPosition00 = ivec2(bucketPosition);
  ivec2 bucketPosition10 = bucketPosition00 + ivec2(xOffset, 0);
  ivec2 bucketPosition01 = bucketPosition00 + ivec2(0, yOffset);
  ivec2 bucketPosition11 = bucketPosition00 + ivec2(xOffset, yOffset);

  ivec2 bucketNum = ivec2(simulationSpace / u_viewRadius) + 1;

  float sum = 0.0;
  sum += findNeighbors(position, bucketPosition00, bucketNum);
  sum += findNeighbors(position, bucketPosition10, bucketNum);
  sum += findNeighbors(position, bucketPosition01, bucketNum);
  sum += findNeighbors(position, bucketPosition11, bucketNum);

  // v_color = vec3(float(bucketPosition00.x * bucketNum.x + bucketPosition00.y) / float(bucketNum.x * bucketNum.y));
  v_color = vec3(sum / 500.0, 0.5, 0.5);
}
`

  const RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_color;

out vec4 o_color;

void main(void) {
  if (length(gl_PointCoord - 0.5) < 0.5) {
    o_color = vec4(v_color, 1.0);
  } else {
    discard;
  }
}
`

const INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out uvec2 o_particle;

uniform sampler2D u_positionTexture;
uniform float u_viewRadius;

float simulationSpace = 1.0;

uint convertCoordToIndex(uvec2 coord, uvec2 size) {
  return coord.x + size.x * coord.y;
}

uint getBucketIndex(vec2 position) {
  uvec2 bucketCoord = uvec2(position / u_viewRadius);
  uvec2 bucketNum = uvec2(simulationSpace / u_viewRadius) + 1u;
  return bucketCoord.x + bucketCoord.y * bucketNum.x;
}

void main(void) {
  vec2 position = texelFetch(u_positionTexture, ivec2(gl_FragCoord.xy), 0).xy;
  uint particleIndex = convertCoordToIndex(uvec2(gl_FragCoord.xy), uvec2(textureSize(u_positionTexture, 0)));
  uint bucketIndex = getBucketIndex(position);
  o_particle = uvec2(bucketIndex, particleIndex);
}

`

  const SWAP_BUCKET_INDEX_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;
precision highp usampler2D;

out uvec2 o_bucket;

uniform usampler2D u_bucketTexture;
uniform uint u_size;
uniform uint u_blockStep;
uniform uint u_subBlockStep;

uint convertCoordToIndex(uvec2 coord) {
  return coord.x + coord.y * u_size;
}

uvec2 convertIndexToCoord(uint index) {
  return uvec2(index % u_size, index / u_size);
}

void main(void) {
  uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy));
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

  uvec2 a = texelFetch(u_bucketTexture, ivec2(gl_FragCoord.xy), 0).xy;
  uvec2 b = texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(targetIndex)), 0).xy;

  if (a.x == b.x) {
    o_bucket = a; // no swap
  } else if ((a.x >= b.x) == up) {
    o_bucket = b; // swap
  } else {
    o_bucket = a; // no swap
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

int convertCoordToIndex(ivec2 coord, ivec2 size) {
  return coord.x + coord.y * size.x;
}

ivec2 convertIndexToCoord(int index, ivec2 size) {
  return ivec2(index % size.x, index / size.x);
}


int getBucketIndex(int particleIndex) {
  ivec2 particleTextureSize = textureSize(u_bucketTexture, 0);
  return int(texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(particleIndex, particleTextureSize)), 0).x);
}


int binarySearchMinIndex(int target, int from, int to) {
  for (int i = 0; i < u_particleNumN + 1; i++) {
    int middle = from + (to - from) / 2;
    int bucketIndex = getBucketIndex(middle);
    if (bucketIndex < target) {
      from = middle + 1;
    } else {
      to = middle;
    }
    if (from == to) {
      if (getBucketIndex(from) == target) {
        return from;
      } else {
        return -1;
      }
    }
  }
  return -1;
}

int binarySearchMaxIndex(int target, int from, int to) {
  for (int i = 0; i < u_particleNumN + 1; i++) {
    int middle = from + (to - from) / 2 + 1;
    int bucketIndex = getBucketIndex(middle);
    if (bucketIndex > target) {
      to = middle - 1;
    } else {
      from = middle;
    }
    if (from == to) {
      if (getBucketIndex(from) == target) {
        return from;
      } else {
        return -1;
      }
    }
  }
  return -1;
}

ivec2 binarySearchRange(int target, int from, int to) {
  from =  binarySearchMinIndex(target, from, to);
  to = from == -1 ? -1 : binarySearchMaxIndex(target, from, to);
  return ivec2(from, to);
}

void main(void) {
  int bucketIndex = convertCoordToIndex(ivec2(gl_FragCoord.xy), u_bucketReferrerTextureSize);
  ivec2 bucketNum = ivec2(simulationSpace / u_viewRadius) + 1;
  int maxBucketIndex = bucketNum.x * bucketNum.y;

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


  function createInitializeParticlesProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, INITIALIZE_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createUpdateParticlesProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createRenderParticlesProgram(gl) {
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
    const positionTexture = createTexture(gl, size, gl.RG32F, gl.RG, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const velocityTexture = createTexture(gl, size, gl.RG32F, gl.RG, gl.FLOAT);
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

  const initializeParticleProgram = createInitializeParticlesProgram(gl);
  const updateParticleProgram = createUpdateParticlesProgram(gl);
  const renderParticleProgram = createRenderParticlesProgram(gl);
  const initializeBucketProgram = createInitializeBucketProgram(gl);
  const swapBucketIndexProgram = createSwapBucketIndexProgram(gl);
  const initializeBucketReferrerProgram = createInitializeBucketReferrerProgram(gl);
  const debugBitonicSortProgram = createDebugBitonicSortProgram(gl);
  const debugBucketReferrerProgram = createDebugBucketReferrerProgram(gl);

  const initializeParticleUniforms = getUniformLocations(gl, initializeParticleProgram, ['u_randomSeed']);
  const updateParticleUniforms = getUniformLocations(gl, updateParticleProgram, ['u_positionTexture', 'u_velocityTexture', 'u_deltaTime']);
  const renderParticleUniforms = getUniformLocations(gl, renderParticleProgram, ['u_positionTexture', 'u_bucketTexture', 'u_bucketReferrerTexture', 'u_canvasSize', 'u_viewRadius']);
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

  const particleTextureSizeN = 7;
  const particleTextureSize = 2**particleTextureSizeN;
  const particleNumN = particleTextureSizeN * 2;
  const particleNum = particleTextureSize * particleTextureSize;

  const viewRadius = 0.1;
  const bucketSize = Math.ceil(1.0 / viewRadius);
  let bucketReferrerTextureSize;
  for (let i = 0; ; i++) {
    bucketReferrerTextureSize = 2**i;
    if (bucketReferrerTextureSize > bucketSize) {
      break;
    }
  }

  const particleVao = createVao(gl,[{
    buffer: createVbo(gl, new Float32Array(Array.from({length: particleNum}, (v, i) => i))),
    size: 1,
    index: 0
  }]);

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
    // gl.uniform2f(initializeParticleUniforms['u_randomSeed'], 100.0, 100.0);
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

  const constructBucketIndex = function() {
    initializeBucket();

    // sort by bitonic sort
    for (let i = 0; i < particleNumN; i++) {
      for (let j = 0; j <= i; j++) {
        swapBucketIndex(i, j);
      }
    }
    
    initializeBucketRefrrer();
  }

  const findNeighbors = function() {
    constructBucketIndex();
  }

  const updateParticles = function(deltaTime) {

    // gl.bindFramebuffer(gl.FRAMEBUFFER, particleFbObjW.framebuffer);
    // gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);

    // gl.useProgram(updateParticleProgram);
    // setTextureAsUniform(gl, 0, particleFbObjR.positionTexture, updateParticleUniforms['u_positionTexture']);
    // setTextureAsUniform(gl, 1, particleFbObjR.velocityTexture, updateParticleUniforms['u_velocityTexture']);
    // gl.uniform1f(updateParticleUniforms['u_deltaTime'], deltaTime);

    // gl.bindVertexArray(fillScreenVao);
    // gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    // gl.bindVertexArray(null);

    // gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // swapParticleFbObj();

    findNeighbors();
  };


  initializeParticles();

  gl.clearColor(0.2, 0.2, 0.2, 1.0);
  let previousTime = performance.now();
  const render = function() {
    stats.update();

    const currentTime = performance.now();
    const deltaTime = Math.min(0.05, (currentTime - previousTime) * 0.001);
    previousTime = currentTime;

    updateParticles(deltaTime);

    gl.viewport(0.0, 0.0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);


    gl.useProgram(renderParticleProgram);
    setTextureAsUniform(gl, 0, particleFbObjR.positionTexture, renderParticleUniforms['u_positionTexture']);
    setTextureAsUniform(gl, 1, bucketFbObjR.bucketTexture, renderParticleUniforms['u_bucketTexture']);
    setTextureAsUniform(gl, 2, bucketReferrerFbObj.bucketReferrerTexture, renderParticleUniforms['u_bucketReferrerTexture']);
    gl.uniform2f(renderParticleUniforms['u_canvasSize'], canvas.width, canvas.height);
    gl.uniform1f(renderParticleUniforms['u_viewRadius'], viewRadius);

    gl.bindVertexArray(particleVao);
    gl.drawArrays(gl.POINTS, 0, particleNum);
    gl.bindVertexArray(null);


    // gl.useProgram(debugBitonicSortProgram);
    // setTextureAsUniform(gl, 0, bucketFbObjR.bucketTexture, debugBitonicSortUniforms['u_bucketTexture']);
    // gl.uniform1ui(debugBitonicSortUniforms['u_bucketNum'], bucketSize * bucketSize);
    // gl.uniform2f(debugBitonicSortUniforms['u_canvasSize'], canvas.width, canvas.height);

    // gl.bindVertexArray(fillScreenVao);
    // gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    // gl.bindVertexArray(null);



    // gl.useProgram(debugBucketReferrerProgram);
    // setTextureAsUniform(gl, 0, bucketReferrerFbObj.bucketReferrerTexture, debugBucketReferrerUniforms['u_bucketReferrerTexture']);
    // gl.uniform1ui(debugBucketReferrerUniforms['u_particleNum'], particleNum);
    // gl.uniform2f(debugBucketReferrerUniforms['u_canvasSize'], canvas.width, canvas.height);

    // gl.bindVertexArray(fillScreenVao);
    // gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    // gl.bindVertexArray(null);

    requestAnimationFrame(render);
  };
  render();

}());