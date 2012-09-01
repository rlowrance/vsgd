-- vsgdLd-test.lua
-- unit test of vsgdLd.lua

require 'vsgdLd'

test = {}
tester = torch.Tester()

-- assert norm(expected - actual) is small
local function assertEqTensor(actual, expected1, expected2, tol)
   local tol = tol or 0.1
   local expected = torch.Tensor(2)
   expected[1] = expected1
   expected[2] = expected2
   print('expected', expected)
   print('actual', actual)
   print('expected-actual', expected - actual)
   print('[1] expected, actual', expected[1], actual[1])
   print('norm diff', torch.norm(expected - actual))
   tester:assertlt(torch.norm(expected - actual), tol, 
                   'expected-actual=' .. tostring(expected-actual))
end

-- function is from Heath, Scientific Computing, p. 282
-- f(x1, x2) = 0.5 x_1^2 + 2.5 x_2^2
-- the minizer is at the origin [0 0]^T
-- return f(x), df/dx, and df^2/dx^2
local Heath = torch.class('Heath')

function Heath.__init()
   self.nextSampleIndex = 1
   self.samples = torch.Tensor({{1, 1},
                                {2, 2},
                                {3, 3}})
   self.nSamples = self.samples:size(1)
   self.nDimensions = self.samples:size(2)
end

function Heath.nDimensions()
   return self.nDimensions
end

function Heath.run()
   local sample = self.samples[self.nextSampleIndex]
   if self.nextSampleIndex = self.nSamples then
      self.nextSampleIndex = 1
   else
      self.nextSampleIndex = self.nextSampleIndex + 1
   end

   local x1 = sample[1]
   local x2 = sample[2]

   local g = torch.Tensor(2)
   g[1] = x1
   g[2] = 5 * x2

   local h = torch.Tensor(2)
   h[1] = 1
   h[2] = 5

   return g, h
end

-- function from Heath with contrived epsilon and c values
-- results were computed by hand
function test.heathTestKnownResults()
   local trace = true
   if trace then print('\n') end

   local heath = Heath()
   local d = heath.nDimensions()

   local vsgd = Vsgd()

   state = {}
   state.epsilon = 1e-10
   state.nSamples = heathSamples:size(1)
   state.n0 = state.nSamples         -- init with all samples
   state.c = 6

   -- initial starting point
   local theta = torch.Tensor(d):fill(0.1)
   if trace then print('state before call', state) end
   local theta, seq = vsgd:ld(heath.run, theta, state)
   if trace then print('state after call', state) end

   assertEqTensor(state.g, 1 + 2/3, 8 + 1/3)
   assertEqTensor(state.v, 19, 475)
   assertEqTensor(state.h, 4.3333, 21.6667)
   local tolerance = 1e-6
   tester:assertle(math.abs(state.eta - 0.0070) < tolerance,
                   'state.eta=' .. state.eta)
   assertEqTensor(state.tau, 3.5614, 3.5614)
   print('theta', theta)
   local tol = 1 -- value are about 10^9
   assertEqTensor(theta, 0.0438, 0.0975, tol)
   tester:asserteq(1, #seq, '1 element')
   tester:asserteq(3, seq[1], 'value should be 3')
   halt()
end

-- function from Heath, attempt to find its minimizer, which is [0 0]
function test.heathMinimizer()
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
   for i = 1, 100 do
      theta, seq = vsgdLd(heathOpfunc3, theta, state)
      if false then
         print('theta', theta)
         print('seq', seq)
         print('state', state)
      end
      print(string.format('%d theta [%f, %f]', i, theta[1], theta[2]))
   end
   assertEqTensor(theta, 0, 0)
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
   tester:add(test.heathTestKnownResults, 'test.heathTestKnownResults')
   --tester:add(test.heathMinimizer, 'test.heathMinimizer')
else
   test:add(test)
end
tester:run()

