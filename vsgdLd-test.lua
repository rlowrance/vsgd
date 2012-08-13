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
do 
   local function heath(x)
      local x1 = x[1]
      local x2 = x[2]
      return 0.5 * x1 * x1 + 2.5 * x2 * x2
   end
   
   heathSamples = torch.Tensor({{1, 1},
                                {2, 2},
                                {3, 3}})
   
   function heathOpfunc3(j)
      assert(1 <= j and j <= heathSamples:size(1))
      local x = heathSamples[j]
      local x1 = x[1]
      local x2 = x[2]
      
      local f = heath(x)
      
      local d = heathSamples:size(2)

      local gradient = torch.Tensor(d)
      gradient[1] = x1
      gradient[2] = 5 * x2
      
      local hessdiag = torch.Tensor(d)
      hessdiag[1] = 1
      hessdiag[2] = 5
      
      return f, gradient, hessdiag
   end
end


-- function from Heath with contrived epsilon and c values
-- results were computed by hand
function test.heathTestKnownResults()
   local trace = true
   if trace then print('\n') end

   state = {}
   state.epsilon = 1e-10
   state.nSamples = heathSamples:size(1)
   state.n0 = state.nSamples         -- init with all samples
   state.c = 6

   local d = heathSamples:size(2)

      
   local theta = torch.Tensor(d):fill(0.1)
   state.inOrder = true
   for i = 1, 3 do
      if trace then
         print('state before call sgdSZL', state)
         print('theta before call sgdSZL', theta)
      end

      local theta, seq = sgdSZL(heathOpfunc3, theta, state)

      if trace then
         print('i', i)
         print('state after call sgdSZL', state)
         print('theta after call sgdSZL', theta)
         print('seq', seq)
      end

      if i == 1 then
         assertEqTensor(state.g, 1 + 2/3, 8 + 1/3)
         assertEqTensor(state.v, 19, 475)
         assertEqTensor(state.h, 4.3333, 21.6667)
         assertEqTensor(state.eta, 0.0337, 0.0003)
         assertEqTensor(state.tau, 3.5614, 3.5614)
         print('theta', theta)
         local tol = 1 -- value are about 10^9
         assertEqTensor(theta, 0.0438, 0.0975 x, tol)
         tester:asserteq(1, #seq, '1 element')
         tester:asserteq(3, seq[1], 'value should be 3')
      end
   end
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
   for i = 1, 25 do
      theta, seq = sgdSZL(heathOpfunc3, theta, state)
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
      theta, seq = sgdSZL(rosenbrockOpfunc3, theta, state)
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

oneTest = test.heathTestKnownResults
if oneTest then
   for k, v in oneTest do
      tester:add({v})
      tester:run()
   end
else
   tester:add(test)
   tester:run()
end
