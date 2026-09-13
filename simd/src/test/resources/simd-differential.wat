(module
  (memory (export "memory") 1)
  ;; constant
  (func (export "v128.const") (result v128)
    v128.const i8x16 0 -1 -128 127 1 -2 126 -127 0 -1 -128 127 1 -2 126 -127)
  ;; load
  (func (export "v128.load") (param i32) (result v128)
    local.get 0
    v128.load)
  ;; store
  (func (export "v128.store") (param i32) (param v128)
    local.get 0
    local.get 1
    v128.store)
  ;; bitwise operations
  (func (export "v128.and") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    v128.and)
  (func (export "v128.or") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    v128.or)
  (func (export "v128.xor") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    v128.xor)
  (func (export "v128.not") (param v128) (result v128)
    local.get 0
    v128.not)
  ;; lane arithmetic
  (func (export "i8x16.add") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    i8x16.add)
  (func (export "i8x16.sub") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    i8x16.sub)
  (func (export "i16x8.add") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    i16x8.add)
  (func (export "i32x4.add") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    i32x4.add)
  (func (export "i32x4.sub") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    i32x4.sub)
  (func (export "i32x4.mul") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    i32x4.mul)
  (func (export "i64x2.add") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    i64x2.add)
  (func (export "i64x2.sub") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    i64x2.sub)
  ;; shifts
  (func (export "i64x2.shl") (param v128) (param i32) (result v128)
    local.get 0
    local.get 1
    i64x2.shl)
  (func (export "i64x2.shr_u") (param v128) (param i32) (result v128)
    local.get 0
    local.get 1
    i64x2.shr_u)
  ;; lane extraction
  (func (export "i32x4.extract_lane.0") (param v128) (result i32)
    local.get 0
    i32x4.extract_lane 0)
  (func (export "i32x4.extract_lane.1") (param v128) (result i32)
    local.get 0
    i32x4.extract_lane 1)
  (func (export "i32x4.extract_lane.2") (param v128) (result i32)
    local.get 0
    i32x4.extract_lane 2)
  (func (export "i32x4.extract_lane.3") (param v128) (result i32)
    local.get 0
    i32x4.extract_lane 3)
  ;; shuffle
  (func (export "i8x16.shuffle") (param v128) (param v128) (result v128)
    local.get 0
    local.get 1
    i8x16.shuffle 0 17 2 19 4 21 6 23 8 25 10 27 12 29 14 31)
)
