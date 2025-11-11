import React from 'react';
import 'react-native-gesture-handler';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'react-native';

// Import all screens
import LocationSearchScreen from './src/screens/LocationSearchScreen';
import NavigationScreen from './src/screens/NavigationScreen';
import Login from './src/screens/Login';
import Register from './src/screens/Register';
import Quiz from './src/screens/Quiz';
import Home from './src/screens/Home';
import Profile from './src/screens/Profile';
import Plan from './src/screens/Plan';
import Journey from './src/screens/Journey';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <>
      <StatusBar barStyle="light-content" backgroundColor="#2196F3" />
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Login"
          screenOptions={{
            headerStyle: {
              backgroundColor: '#2196F3',
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
              fontSize: 18,
            },
            headerBackTitleVisible: false,
          }}
        >
          {/* Auth Screens */}
          <Stack.Screen
            name="Login"
            component={Login}
            options={{
              title: 'Welcome to NavTwin',
              headerShown: false,
            }}
          />

          <Stack.Screen
            name="Register"
            component={Register}
            options={{
              title: 'Create Account',
              headerShown: false,
            }}
          />

          {/* Quiz Screen */}
          <Stack.Screen
            name="Quiz"
            component={Quiz}
            options={{
              title: 'Sensory Profile Quiz',
              headerShown: false,
            }}
          />

          {/* Main App Screens */}
          <Stack.Screen
            name="Home"
            component={Home}
            options={{
              title: 'NavTwin',
              headerLeft: null, // Prevent going back to login
              gestureEnabled: false,
            }}
          />

          <Stack.Screen
            name="Profile"
            component={Profile}
            options={{
              title: 'My Profile',
            }}
          />

          <Stack.Screen
            name="LocationSearch"
            component={LocationSearchScreen}
            options={{
              title: 'Plan Your Journey',
              headerShown: false,
            }}
          />

          <Stack.Screen
            name="Journey"
            component={Journey}
            options={{
              title: 'Active Journey',
              headerShown: false,
              gestureEnabled: false,
            }}
          />

          <Stack.Screen
            name="Navigation"
            component={NavigationScreen}
            options={{
              title: 'Navigation',
              headerShown: false,
              gestureEnabled: false,
            }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </>
  );
}