-- Vsgdd-test.lua
-- unit test of Vsgd.lua

require 'Vsgd'

test = {}
tester = torch.Tester()

-- assert norm(expected - actual) is small
local function assertEqTensor(actual, expected1, expected2, tol)
   local trace = false
   local tol = tol or 0.1
   local expected = torch.Tensor(2)
   expected[1] = expected1
   expected[2] = expected2
   if trace then
      print('expected', expected)
      print('actual', actual)
      print('expected-actual', expected - actual)
      print('norm diff', torch.norm(expected - actual))
   end
   local diff = torch.norm(expected - actual)
   tester:assertle(diff, tol, 
                   'expected-actual=' .. tostring(expected-actual))
   if false and not (diff <= tol) then
      error('stopping, since assertion failed; diff = ' .. diff .. 
            ' tol = ' .. tol)
   end
end

-- function is from Heath, Scientific Computing, p. 282
-- f(x1, x2) = 0.5 x_1^2 + 2.5 x_2^2
-- the minizer is at the origin [0 0]^T
do
   -- must hide the definition of Heath for the non-class code
   local trace = false
   local Heath = torch.class('Heath')

   function Heath:__init()
      self._nextSampleIndex = 1
      self._samples = torch.Tensor({{1, 1},
                                   {2, 2},
                                   {3, 3}})
      self._nSamples = self._samples:size(1)
      self._nDimensions = self._samples:size(2)
   end

   function Heath:nDimensions()
      return self._nDimensions
   end

   function Heath:nSamples()
      return self._nSamples
   end

   -- return fx, df/dx, and df^2/dx^2
   function Heath:run()
      if trace then
         print('\nHeath:run self', self)
      end

      local sample = self._samples[self._nextSampleIndex]

      if self._nextSampleIndex == self._nSamples then
         self._nextSampleIndex = 1
      else
         self._nextSampleIndex = self._nextSampleIndex + 1
      end

      local x1 = sample[1]
      local x2 = sample[2]

      local fx = 0.5 * x1 * x1 + 2.5 * x2 * x2

      local g = torch.Tensor(2)
      g[1] = x1
      g[2] = 5 * x2

      local h = torch.Tensor(2)
      h[1] = 1
      h[2] = 5

      return fx, g, h
   end
end

function test.Heath() -- test Heath class
   local heath = Heath()
   local function f()
      return heath:run()
   end

   local function check(expectedFx, 
                        expectedG1, expectedG2,
                        j)
      local trace = false
      local actualFx, actualG, actualH = f()
      if trace then 
         print('check fx, g, h', actualFx, actualG, actualH)
      end
      tester:asserteq(expectedFx, actualFx, 'fx j=' .. j)
      assertEqTensor(actualG, expectedG1, expectedG2, 0)
      assertEqTensor(actualH, 1, 5, 0)
   end

   check(3, 1, 5, 1)
   check(12, 2, 10, 2)
   check(27, 3, 15, 3)
end


-- function from Heath with specific epsilon and c values
-- results were computed by hand for one iteration
function test.heathTestKnownResults()
   local trace = false
   if trace then print('\n') end

   local heath = Heath()
   if trace then print('Heath instance', heath) end
   local d = heath:nDimensions()

   local vsgd = Vsgd()

   state = {}
   state.epsilon = 1e-10
   state.nSamples = heath:nSamples()
   state.n0 = state.nSamples         -- init with all samples
   state.c = 6

   -- initial starting point
   local theta = torch.Tensor(d):fill(0.1)
   if trace then print('state before call', state) end
   local function f()
      return heath:run()
   end
   local theta, seq = vsgd:ld(f, theta, state)
   if trace then print('state after call', state) end

   assertEqTensor(state.g, 1 + 2/3, 8 + 1/3)
   assertEqTensor(state.v,  19.0,    475.0)
   assertEqTensor(state.h,   4.3333,  21.6667)
   assertEqTensor(state.eta, 0.0337,   0.0067)
   assertEqTensor(state.tau, 3.5614,   3.5614)
   assertEqTensor(theta,     0.0663,   0.0665)
   tester:asserteq(1, #seq, '1 element')
   tester:asserteq(3, seq[1], 'value should be 3')
end

-- attempt to find its minimizer of the function from Heath
function test.heathMinimizer()
   local trace = true
   if trace then print('\n') end

   local heath = Heath()
   local function f()
      return heath:run()
   end

   local function minimize(startX1, startX2, n0, steps)
      assert(startX1)
      assert(startX2)

      state = {}
      state.epsilon = 1e-10 
      state.nSamples = heath:nSamples()
      if n0 == nil then
         state.n0 = state.nSamples
      else
         state.n0 = n0
      end
      state.c = 6
      
      -- attempt to minimize
      local d = heath:nDimensions()
      local theta = torch.Tensor(2)
      theta[1] = startX1
      theta[2] = startX2
      print(string.format('\n%d theta [%f, %f]', 0, theta[1], theta[2]))
      local vsgd = Vsgd()
      steps = steps or 10
      for i = 1, steps do
         theta, seq = vsgd:ld(f, theta, state)
         print(string.format('%d theta [%f, %f]', i, theta[1], theta[2]))
         if i == 1 then
            -- should be same as for test.heathTestKnownResults
         assertEqTensor(theta, 0.0663, 0.0665) 
         print('state after the first step')
         vsgd:printState(state)
         end
      end
      assertEqTensor(theta, 0, 0)
   end --minimize

   minimize(0.1, 0.1) -- same starting point as for known results test
   minimize(0.1, 0.1, 100, 30)
   minimize(5.0, 1.0) -- as in Heath p. 277 using gradient descent
end

--------------------------------------------------------------------------------
-- Rosenbrock function
--------------------------------------------------------------------------------

do
   local function rosenbrock(tensor)
      local x = tensor[1]
      local y = tensor[2]
      
      local term1 = 1 - x
      local term2 = y - x * x
      
      return term1 * term1 + 100 * term2 * term2
   end

   local d = 2
   local nSamples = 100
   rosenbrockSamples = torch.randn(nSamples, 2)

   function rosenbrockOpfunc3(j)
      assert(1 <= j and j <= rosenbrockSamples:size(1))
      local tensor = rosenbrockSamples[j]
      local x = tensor[1]
      local y = tensor[2]

      local f = rosenbrock(tensor)

      local d = 2
      local gradient = torch.Tensor(d)
      local a =  2 * (1 - x) * (-1)
      local b = 200 * (y - x * x) * (- 2 * x)
      gradient[1] = 2 * (1 - x) * (-1) + 200 * (y - x * x) * (- 2 * x)
      gradient[2] = 200 * (y - x * x)

      local hessdiag = torch.Tensor(d)
      hessdiag[1] = 
         2 * (1 - x) * (-1) * (-1) + 
         200 * ((y - x * x) * (-2) + (-2 * x) * (y - x* x) * (-2 * x))
      hessdiag[2] = 200 * (y - x * x)

      return f, gradient, hessdiag
   end
end

      

-- function from Rosenbrock, attempt to find its minimizer, which is [1 1]
function test.rosenbrockMinimizer()
   local trace = true
   if trace then print('\n') end
   
   state = {}
   state.epsilon = 1e10 -- should not play a role
   state.nSamples = heathSamples:size(1)
   state.n0 = state.nSamples
   state.c = 1

   -- attempt to minimize
   local d = heathSamples:size(2)
   local theta = torch.randn(d)
   for i = 1, 25 do
      theta, seq = vsgdLd(rosenbrockOpfunc3, theta, state)
      if false then
         print('theta', theta)
         print('seq', seq)
         print('state', state)
      end
      print(string.format('%d theta [%f, %f]', i, theta[1], theta[2]))
   end
   assertEqTensor(theta, 1, 1)
end

--------------------------------------------------------------------------------
-- main
--------------------------------------------------------------------------------

if true then
   --tester:add(test.Heath, 'test.Heath')
   --tester:add(test.heathTestKnownResults, 'test.heathTestKnownResults')
   tester:add(test.heathMinimizer, 'test.heathMinimizer')
else
   test:add(test)
end
tester:run()

